try:
    import json_tricks
    json_tricks_loaded = True
except ImportError:
    # Loading and saving of the retriever is not possible
    json_tricks_loaded = False

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVR

__all__ = [
    'RetrievalProduct',
]


class NotTrainedError(Exception):
    """Should be raised if someone runs a non-trained retrieval product
    """
    def __init__(self, *args):
        message = "You must train this retrieval product before running it!"
        Exception.__init__(self, message, *args)


class RetrievalProduct:
    """Retrieval that can be trained with data and stored to json files

    This is basically a wrapper around the scikit-learn estimator and trainer
    classes and makes it possible to save the trained models as json files.

    """

    def __init__(
            self, parameter=None, parameters_file=None, estimator=None,
            trainer=None, verbose=False):
        """Initialize a Retriever object

        Args:
            parameter: A dictionary with training parameters.
            parameters_file: Name of file with previously stored training
                parameters.
            estimator: Object that will be used for predicting the
                retrieval. This object can be a sklearn Estimator or Pipeline
                object.
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

        self.verbose = verbose

        if parameters_file is not None:
            self.load_parameters(parameters_file)

    @staticmethod
    def _default_estimator(estimator):
        """Return the default estimator"""

        # Estimators are normally objects that have a fit and predict method
        # (e.g. MLPRegressor from sklearn). To make their training easier we
        # scale the input data in advance. With Pipeline objects from sklearn
        # we can combine such steps easily since they pretend to be estimator
        # objects as well.
        if estimator is None or estimator.lower() == "nn":
            estimator = MLPRegressor(max_iter=2000)
        elif estimator.lower() == "svr":
            estimator = SVR(kernel="rbf")
        else:
            raise ValueError(f"Unknown estimator type: {estimator}!")

        return Pipeline([
            # SVM or NN work better if we have scaled the data in the first
            # place. MinMaxScaler is the simplest one. RobustScaler or
            # StandardScaler could be an alternative.
            #("scaler", MinMaxScaler(feature_range=[0, 1])),
            ("scaler", StandardScaler()),
            #("scaler", RobustScaler(quantile_range=(25, 75))),
            # The "real" estimator:
            ("estimator", estimator),
        ])

    def _default_trainer(self):
        """Return the default trainer for the current estimator
        """
        if self.estimator is None or isinstance(self.estimator, str):
            estimator = self._default_estimator(self.estimator)
        else:
            estimator = self.estimator

        # To optimize the results, we try different hyper parameters by
        # using a grid search
        if isinstance(estimator.steps[-1][1], MLPRegressor):
            # Hyper parameter for Neural Network
            hidden_layer_sizes = [
                (15, 10, 3,), (15, 5, 3, 5), (15, 10), (15, 3),
            ]
            common = {
                'estimator__activation': ['relu', 'tanh'],
                'estimator__hidden_layer_sizes': hidden_layer_sizes,
                'estimator__random_state': [0, 5, 9],
                #'alpha': 10.0 ** -np.arange(1, 7),
            }
            hyper_parameter = [
                {   # Hyper parameter for lbfgs solver
                    'estimator__solver': ['lbfgs'],
                    **common
                },
                # {  # Hyper parameter for adam solver
                #     'solver': ['adam'],
                #     'batch_size': [200, 1000],
                #     'beta_1': [0.95, 0.99],
                #     'beta_2': [0.95, 0.99],
                #     **common
                # },
            ]
        elif isinstance(estimator.steps[-1][1], SVR):
            # Hyper parameter for Support Vector Machine
            hyper_parameter = {
                "estimator__gamma": 10.**np.arange(-3, 2),
                "estimator__C": 10. ** np.arange(-3, 2),
            }
        else:
            raise ValueError(
                f"No default trainer for {estimator} implemented! Define one "
                f"by yourself via __init__(*args, ?trainer?).")

        return GridSearchCV(
            estimator, hyper_parameter, n_jobs=4,
            refit=True, cv=3, verbose=self.verbose,
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
            "params": model.get_params(deep=True),
            "coefs": {
                name: getattr(model, name)
                for name in model.__dir__()
                if not name.startswith("__") and name.endswith("_")
            }
        }
        return json

    @staticmethod
    def _set_sklearn_coefs(sklearn_obj, coefs):
        for attr, value in coefs.items():
            setattr(sklearn_obj, attr, value)

    def load_parameters(self, filename):
        """ Load the training parameters from a JSON file.

        Training parameters are:
        * weights of the neural networks (classifier and regressor)
        * names of the input and target fields

        Args:
            filename: The name of file from where to load the training
                parameters.

        Returns:
            None
        """

        if not json_tricks_loaded:
            raise ImportError(
                "Could not load the json_tricks module, which is required to "
                "load retrievals from json files."
            )

        with open(filename, 'r') as infile:
            parameter = json_tricks.load(infile)

            estimator = parameter.pop("estimator", None)
            if estimator is None:
                raise ValueError("Found no coefficients for estimator!")

            estimator = self._create_model(
                MLPRegressor, estimator["params"], estimator["coefs"],
            )

            # self._set_sklearn_coefs(self.estimator, estimator_coefs)

            scaler = parameter.pop("scaler", None)
            if scaler is None:
                raise ValueError("Found no coefficients for scaler!")

            scaler = self._create_model(
                RobustScaler, scaler["params"], scaler["coefs"],
            )

            self.estimator = Pipeline([
                ("scaler", scaler,),  ("estimator", estimator)
            ])

            self.parameter.update(parameter)

    def retrieve(self, inputs):
        """Predict the target values for data coming from arrays

        Args:
            inputs: A pandas.DataFrame object The keys must be the
                same labels as used in :meth:`train`. The values are the field
                names in the original data coming from *data*. If not given,
                the same input field names are taken as during training with
                :meth:`train` (or were loaded from a parameter file).

        Returns:
            If output is not None, a :class:`FileSet` object holding the data.
            Otherwise a xarray.Dataset with the retrieved data.

        Examples:

        .. :code-block:: python

            # TODO
        """

        if self.estimator is None:
            raise NotTrainedError()

        # Skip to small datasets
        if inputs.empty:
            return None

        # Retrieve the data from the neural network:
        output_data = self.estimator.predict(inputs)

        retrieved_data = pd.DataFrame()
        target_label = self.parameter["targets"][0]
        retrieved_data[target_label] = output_data

        return retrieved_data

    def save_parameters(self, filename):
        """ Save the training parameters to a JSON file.

        Training parameters are:
            * configuration of the used estimator
            * configuration of the used scaler
            * names of the input, output, and target fields

        Args:
            filename: The name of the file where to store the training
                parameters.

        Returns:
            None
        """

        if not json_tricks_loaded:
            raise ImportError(
                "Could not load the json_tricks module, which is required to "
                "save retrievals to json files."
            )

        parameter = self.parameter.copy()

        parameter["scaler"] = self._model_to_json(
            self.estimator.steps[0][1]
        )
        parameter["estimator"] = self._model_to_json(
            self.estimator.steps[-1][1]
        )

        with open(filename, 'w') as outfile:
            json_tricks.dump(parameter, outfile)

    def score(self, inputs, targets):
        """

        Args:
            data:
            inputs:
            targets:
            metric

        Returns:
            The metric score as a number
        """
        if self.estimator is None:
            raise NotTrainedError()

        return self.estimator.score(inputs, targets.values.ravel())

    def train(self, inputs, targets, verbose=0):
        """Train this retriever with data from arrays

        Args:
            inputs: A pandas.DataFrame with input data.
            targets: A pandas.DataFrame with training data.
            verbose: Level of verbosity (=number of debug messages). Default is
                0.

        Returns:
            A float number indicating the training score.
        """

        if self.trainer is None:
            self.trainer = self._default_trainer()

        # The input and target labels will be saved because they can be used in
        # other methods as well.
        self.parameter["inputs"] = inputs.columns.tolist()
        self.parameter["targets"] = targets.columns.tolist()

        # Unleash the trainer
        self.trainer.verbose = verbose
        self.trainer.fit(inputs, targets.values.ravel())

        # Use the best estimator from now on:
        self.estimator = self.trainer.best_estimator_

        return self.estimator.score(inputs, targets)

    def _get_inputs(self, data, fields):

        if fields is None:
            fields = self.parameter["inputs"]

        # Prepare the input data (we need the original order):
        input_data = np.asmatrix([
            data[field]
            for _, field in sorted(fields.items())
        ]).T

        return input_data

    def _get_targets(self, data, fields):

        if fields is None:
            fields = self.parameter["targets"]

        # Prepare the target data:
        target_data = np.asmatrix([
            data[field]
            for _, field in sorted(fields.items())
        ]).T

        return target_data

