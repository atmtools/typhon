import json

import numpy as np
from scipy import stats
from sklearn.model_selection import GridSearchCV, train_test_split, \
    RandomizedSearchCV
import sklearn.neural_network as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from typhon.spareice.array import ArrayGroup
from typhon.spareice.datasets import Dataset, DataSlider

__all__ = [
    'Retriever',
]


class Retriever:
    """Still under development.

    """

    def __init__(
            self, parameter=None, parameters_file=None, model=None,
            scaler=None):
        """

        Either a file with previously stored training parameters must given or
        the training parameters must be set by oneself.

        Args:
            parameter: A dictionary with training parameters.
            parameters_file: Name of file with previously stored training
                parameters.
            model: Object that will be used for learn and produce the
                retrieval. Default is a GridSearchCV with MLPRegressor.
        """

        default_parameter = {
            "inputs": {},
            "targets": {},
            "scaling": None,
        }

        if parameter is None:
            parameter = {}

        self.parameter = {**default_parameter, **parameter}

        # The learners for this Retriever:
        if model is None:
            self.model = self._init_default_model()
        else:
            self.model = model

        if parameters_file is not None:
            self.load_parameters(parameters_file)

        # The data must be scaled before passed to the regressor. Always the
        # same scaling must be applied.
        self._scaler = None
        if self.parameter["scaling"] is not None:
            self._scaler = StandardScaler()
            self._scaler.scale_ = [
                np.array(scaling)
                for scaling in parameter["scaling"]
            ]

    def _init_default_model(self,):
        # To optimize the results, we try different hyper parameters by
        # using the default tuner
        hidden_layer_sizes = [
            (15, 5, 3, 5),  (15, 5, 3,), (15, 5), (15, 10), (15,),
        ]
        hyper_parameter = [
            {   # Hyper parameter for lbfgs solver
                'solver': ['lbfgs'],
                'activation': ['relu',],
                'hidden_layer_sizes': hidden_layer_sizes,
                'random_state': [0, 5, 9],
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
        estimator = nn.MLPRegressor(
            max_iter=1000,
            # verbose=True,
        )

        return GridSearchCV(
            estimator, hyper_parameter, n_jobs=10,
            refit=True, cv=3, verbose=2,
        )

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
            parameter = json.load(infile)

            try:
                self._regressor.coefs_ = [
                    np.array(weights)
                    for weights in parameter["regressor_weights"]
                ]
                self._regressor.intercepts_ = [
                    np.array(biases)
                    for biases in parameter["regressor_biases"]
                ]
            except KeyError:
                pass

            self.parameter.update(parameter)

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

        # try:
        #     self.parameter["classifier_weights"] = self._classifier.coefs_
        # except KeyError:
        #     pass

        self.parameter["regressor_weights"] = \
            [weights.tolist() for weights in self._regressor.coefs_]
        self.parameter["regressor_biases"] = \
            [biases.tolist() for biases in self._regressor.intercepts_]

        if self._scaler is not None:
            self.parameter["scaling"] = \
                [scaling.tolist() for scaling in self._scaler.scale_]

        with open(filename, 'w') as outfile:
            json.dump(self.parameter, outfile)

    def run(self, start, end, sources, inputs, output=None, extra_fields=None,
            cleaner=None):
        """Retrieve the data between two dates

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            sources: Sources where the data come from. Must be a list / dict of
                Dataset or dict-like objects with numpy arrays.
            inputs: A dictionary of input field names. The keys must be the
                same labels as used in :meth:`train`. The values are the field
                names in the original data coming from *sources*.
            output: Either None (default), a path as string containing
                placeholders or a Dataset-like object. If None, all data will
                be returned as one object.
            extra_fields: Extra fields that should be copied to output. If you
                want to save the output to a Dataset, this must contain a
                *time* field.
            cleaner: A filter function that can be used to clean the input
                data.

        Returns:
            If output is not None, a :class:`Dataset` object holding the data.
            Otherwise an ArrayGroup / xarray.Dataset with the retrieved data.

        Examples:

        .. :code-block:: python

            # TODO
        """

        if output is None or isinstance(output, Dataset):
            pass
        elif isinstance(output, str):
            output = Dataset(path=output, name="RetrievedData")
        else:
            raise ValueError("The parameter output must be None, a string or "
                             "a Dataset object!")

        if isinstance(sources, dict):
            slider = DataSlider(start, end, **sources)
        else:
            slider = DataSlider(start, end, *sources)

        results = []

        # Slide through all input sources and apply the regression on them
        for data in map(ArrayGroup.from_dict, slider):
            if callable(cleaner):
                data = data[cleaner(data)]

            input_data = np.asmatrix([
                data[field]
                for _, field in sorted(inputs.items())
            ]).T

            # Skip to small datasets
            if not input_data.any():
                print("Skip this data!")
                continue

            # Scale the input data:
            input_data = self._scaler.transform(input_data)

            # Retrieve the data from the neural network.
            output_data = self.model.predict(input_data)

            print(output_data)

            retrieved_data = ArrayGroup.from_dict({
                name: output_data[:, i]
                for i, name in enumerate(sorted(self.parameter["targets"]))
            })

            print(retrieved_data)

            if extra_fields is not None:
                for new_name, old_name in extra_fields.items():
                    retrieved_data[new_name] = data[old_name]

            if output is None:
                results.append(retrieved_data)
            else:
                # Store the generated data.
                times = retrieved_data["time"].min(), retrieved_data[
                    "time"].max()
                output.write(retrieved_data, times=times, in_background=True)

        return ArrayGroup.concatenate(results)

    def train(self, start, end, sources, inputs, targets, cleaner=None,
              test_size=None):
        """Train this retriever

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            sources: Sources where the data come from. Must be a list / dict of
                Dataset or dict-like objects with numpy arrays.
            inputs: A dictionary of input field names. The keys are labels of
                the input fields. The values are the field names in the
                original data coming from *sources*.
            targets: A dictionary of target field names. The keys are labels of
                the target fields. The values are the field names in the
                original data coming from *sources*.
            cleaner: A filter function that can be used to clean the training
                data.
            verbose: Print debug messages if true.

        Returns:
            Last training and testing score.
        """

        # The input and target labels will be saved because they can be used in
        # run as well.
        self.parameter["inputs"] = inputs
        self.parameter["targets"] = targets

        if isinstance(sources, dict):
            slider = DataSlider(start, end, **sources)
        else:
            slider = DataSlider(start, end, *sources)

        # Get all the data at once:
        data = ArrayGroup.from_dict(slider.flush())

        train_input, test_input, train_target, test_target = \
            self._prepare_training_data(
                data, inputs, targets, cleaner, test_size
            )

        # Train the model
        self.model.fit(train_input, train_target)

        return (
            train_input.shape[0],
            self.model.score(train_input, train_target),
            self.model.score(test_input, test_target)
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
        if self._scaler is None:
            self._scaler = MinMaxScaler(feature_range=[0, 1])
            self._scaler.fit(input_data)

        input_data = self._scaler.transform(input_data)

        return train_test_split(input_data, target_data, test_size=test_size)
