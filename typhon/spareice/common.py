try:
    import json_tricks
except ImportError:
    # Loading and saving of the retriever is not possible
    pass

import numpy as np
from scipy import stats
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typhon.spareice.array import GroupedArrays
from typhon.spareice.datasets import Dataset, DataSlider

__all__ = [
    'Retriever',
]


class Retriever:
    """Retrieval that can be trained with data

    """

    def __init__(
            self, parameter=None, parameters_file=None, estimator=None,
            trainer=None, scaler=None):
        """Initialize a Retriever object

        Args:
            parameter: A dictionary with training parameters.
            parameters_file: Name of file with previously stored training
                parameters.
            estimator: Object that will be used for producing the
                retrieval.
            trainer: Object that will be used to train and to find the best
                retrieval estimator. Default is a GridSearchCV with a
                MLPRegressor.
        """

        default_parameter = {
            "inputs": {},
            "targets": {},
            "scaling": None,
        }

        if parameter is None:
            parameter = {}

        self.parameter = {**default_parameter, **parameter}

        # The trainer and/or model for this retriever:
        self.estimator = estimator
        self.trainer = trainer
        self.scaler = scaler

        if parameters_file is not None:
            self.load_parameters(parameters_file)

    @staticmethod
    def _default_estimator():
        return MLPRegressor(
            max_iter=2000,
            # verbose=True,
        )

    @staticmethod
    def _default_scaler():
        return MinMaxScaler(
            feature_range=[0, 1]
        )

    def _default_trainer(self,):
        # To optimize the results, we try different hyper parameters by
        # using the default tuner
        hidden_layer_sizes = [
            (15, 10, 3,), (15, 5, 3,), (15, 5), (15, 10),
        ]
        hyper_parameter = [
            {   # Hyper parameter for lbfgs solver
                'solver': ['lbfgs'],
                'activation': ['relu', 'tanh'],
                'hidden_layer_sizes': hidden_layer_sizes,
                'random_state': [0, 1, 2, 4, 5, 9],
            },
            # {  # Hyper parameter for adam solver
            #     'solver': ['adam'],
            #     'activation': ['relu', 'tanh', 'logistic'],
            #     'hidden_layer_sizes': hidden_layer_sizes,
            #     'batch_size': [200, 1000],
            #     'beta_1': [0.95, 0.99],
            #     'beta_2': [0.95, 0.99],
            # },
        ]

        if self.estimator is None:
            estimator = self._default_estimator()
        else:
            estimator = self.estimator

        return GridSearchCV(
            estimator, hyper_parameter, n_jobs=10,
            refit=True, cv=3, verbose=2,
        )

    @staticmethod
    def _create_model(model_class, params, coefs):
        model = model_class(**params)
        for attr, value in coefs.items():
            setattr(model, attr, value)

        return model

    @staticmethod
    def _model_to_json(model):
        """

        Returns:
            A tuple with the weights and biases.
        """

        if model is None:
            raise ValueError("No object trained!")

        json = {
            "params": model.get_params(deep=True)
        }
        json["coefs"] = {
            name: getattr(model, name)
            for name in model.__dir__()
            if not name.startswith("__") and name.endswith("_")
        }
        return json
        # return {
        #     name: #[layer.tolist() for layer in attr]
        #     if hasattr(attr, "tolist") else attr
        #     for name, attr in attrs.items()
        # }

    @staticmethod
    def _set_sklearn_coefs(sklearn_obj, coefs):
        for attr, value in coefs.items():
            setattr(sklearn_obj, attr, value)

    def load_parameters(self, filename):
        """ Loads the training parameters from a JSON file.

        Training parameters are:
        * weights of the neural networks (classifier and regressor)
        * names of the input and target fields

        Args:
            filename: The name of file from where to load the training
                parameters.

        Returns:
            None
        """

        with open(filename, 'r') as infile:
            parameter = json_tricks.load(infile)

            estimator = parameter.pop("estimator", None)
            if estimator is None:
                raise ValueError("Found no coefficients for estimator!")

            self.estimator = self._create_model(
                MLPRegressor, estimator["params"], estimator["coefs"],
            )

            # self._set_sklearn_coefs(self.estimator, estimator_coefs)

            scaler = parameter.pop("scaler", None)
            if scaler is None:
                raise ValueError("Found no coefficients for scaler!")

            # The data must be scaled before passed to the regressor. Always
            # the same scaling must be applied.
            self.scaler = self._create_model(
                MinMaxScaler, scaler["params"], scaler["coefs"],
            )

            self.parameter.update(parameter)

    @staticmethod
    def calc_mfe(retrieved, real):
        """Calculate the median fraction error

        Args:
            retrieved: The retrieved data.
            real: The real data that should have been retrieved.

        Returns:
            Median fraction error and bin edges
        """
        fe = np.abs(np.exp(np.log(retrieved / real))) - 1
        mfe, bin_edges, _ = stats.binned_statistic(
            np.log10(real),
            values=fe, statistic="median", bins=20
        )
        return mfe, bin_edges

    def predict(self, data_matrix, scale=True):
        if scale:
            # Scale the input data:
            data_matrix = self.scaler.transform(data_matrix)

        # Retrieve the data from the neural network.
        return self.estimator.predict(data_matrix)

    def run(self, sources, inputs, extra_fields=None, cleaner=None):
        """Predict the target values for data coming from arrays

        Args:
            sources: Sources where the data come from. Must be a dict of
                dict-like objects (such as xarray.Dataset) with numpy arrays.
            inputs: A dictionary of input field names. The keys must be the
                same labels as used in :meth:`train`. The values are the field
                names in the original data coming from *sources*.
            extra_fields: Extra fields that should be copied to output. If you
                want to save the output to a Dataset, this must contain a
                *time* field.
            cleaner: A filter function that can be used to clean the input
                data.

        Returns:
            If output is not None, a :class:`Dataset` object holding the data.
            Otherwise an GroupedArrays / xarray.Dataset with the retrieved
            data.

        Examples:

        .. :code-block:: python

            # TODO
        """

        if not isinstance(sources, dict):
            raise ValueError("Only a dictionary of arrays is allowed!")

        # We running SPARE-ICE only on arrays:
        data = GroupedArrays.from_dict(sources)

        if callable(cleaner):
            data = data[cleaner(data)]

        input_data = np.asmatrix([
            data[field]
            for _, field in sorted(inputs.items())
        ]).T

        # Skip to small datasets
        if not input_data.any():
            print("Skip this data!")
            return None

        # Retrieve the data from the neural network:
        output_data = self.predict(input_data)

        if len(self.parameter["targets"]) == 1:
            retrieved_data = GroupedArrays()
            target_label = list(self.parameter["targets"].keys())[0]
            retrieved_data[target_label] = output_data
        else:
            retrieved_data = GroupedArrays.from_dict({
                name: output_data[:, i]
                for i, name in enumerate(sorted(self.parameter["targets"]))
            })

        if extra_fields is not None:
            for new_name, old_name in extra_fields.items():
                retrieved_data[new_name] = data[old_name]

        return retrieved_data

    def run_datasets(self, sources, start=None, end=None, output=None,
                     inputs=None, extra_fields=None, cleaner=None):
        """Predict the target values for data coming from datasets

        Args:
            sources: List of Dataset objects.
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            output: Either None (default), a path as string containing
                placeholders or a Dataset-like object. If None, all data will
                be returned as one object.
            inputs: A dictionary of input field names. The keys must be the
                same labels as used in :meth:`train`. The values are the field
                names in the original data coming from *sources*.
            extra_fields: Extra fields that should be copied to output. If you
                want to save the output to a Dataset, this must contain a
                *time* field.
            cleaner: A filter function that can be used to clean the input
                data.

        Returns:
            If output is not None, a :class:`Dataset` object holding the data.
            Otherwise an GroupedArrays / xarray.Dataset with the retrieved
            data.
        """

        if output is None or isinstance(output, Dataset):
            pass
        elif isinstance(output, str):
            output = Dataset(path=output, name="RetrievedData")
        else:
            raise ValueError("The parameter output must be None, a string or "
                             "a Dataset object!")

        results = []

        # Slide through all input sources and apply the regression on them
        for files, data in DataSlider(start, end, *sources):
            retrieved_data = self.run(data, inputs, extra_fields, cleaner)

            if retrieved_data is None:
                continue

            if output is None:
                results.append(retrieved_data)
            else:
                # Store the generated data.
                times = retrieved_data["time"].min().item(0), \
                        retrieved_data["time"].max().item(0)

                output.write(retrieved_data, times=times, in_background=True)

        if output is None:
            if results:
                return GroupedArrays.concat(results)
            else:
                return None

    def save_parameters(self, filename):
        """ Saves the training parameters as a JSON file.

        Training parameters are:
            - weights of the neural networks (classifier and regressor)
            - names of the input, output, and target fields

        Args:
            filename: The name of file where to store the training
                parameters.

        Returns:
            None
        """

        parameter = self.parameter.copy()

        parameter["estimator"] = self._model_to_json(self.estimator)
        parameter["scaler"] = self._model_to_json(self.scaler)

        with open(filename, 'w') as outfile:
            json_tricks.dump(parameter, outfile)

    def score(self, inputs, targets, metric=None,):
        return self.trainer.score(inputs, targets)

    def train(self, sources, inputs, targets, cleaner=None, test_size=None,
              verbose=0):
        """Train this retriever with data from arrays

        Args:
            sources: Sources where the data come from. Must be a dictionary of
                dict-like objects (such as xarray.Dataset) with numpy arrays.
            inputs: A dictionary of input field names. The keys are labels of
                the input fields. The values are the field names in the
                original data coming from *sources*.
            targets: A dictionary of target field names. The keys are labels of
                the target fields. The values are the field names in the
                original data coming from *sources*.
            cleaner: A filter function that can be used to clean the training
                data.
            test_size: Fraction of the data should be used for testing not
                training.
            verbose: Level of verbosity (=number of debug messages). Default is
                0.

        Returns:
            The data from *sources* split into training input, testing input,
            training target and testing target.
        """

        if self.trainer is None:
            self.trainer = self._default_trainer()

        # The input and target labels will be saved because they can be used in
        # run as well.
        self.parameter["inputs"] = inputs
        self.parameter["targets"] = targets

        if isinstance(sources, dict):
            data = GroupedArrays.from_dict(sources)
        else:
            raise ValueError("Only a dictionary of arrays is allowed!")

        train_input, test_input, train_target, test_target = \
            self._prepare_training_data(
                data, inputs, targets, cleaner, test_size
            )

        # Unleash the trainer
        self.trainer.verbose = verbose
        self.trainer.fit(train_input, train_target)

        # Use the best estimator from now on:
        self.estimator = self.trainer.best_estimator_

        if verbose:
            print()

        return (
            train_input, test_input, train_target, test_target
        )

    def _prepare_training_data(
            self, data, inputs, targets, cleaner, test_size):
        # Apply the cleaner if there is one
        if callable(cleaner):
            data = data[cleaner(data)]

        # Prepare the input and target data:
        input_data = np.asmatrix([
            data[field]
            for _, field in sorted(inputs.items())
        ]).T
        target_data = np.asmatrix([
            data[field]
            for _, field in sorted(targets.items())
        ]).T

        # Skip too small datasets
        if input_data.shape[0] < 2 or target_data.shape[0] < 2:
            raise ValueError("Not enough data for training!")

        # We have not prepared a scaler yet. This should be done only once.
        if self.scaler is None:
            self.scaler = self._default_scaler()
            self.scaler.fit(input_data)

        input_data = self.scaler.transform(input_data)

        return train_test_split(input_data, target_data, test_size=test_size,
                                shuffle=True)

