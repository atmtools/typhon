import warnings

try:
    import json_tricks
    json_tricks_loaded = True
except ImportError:
    warnings.warn("Loading and saving of the retriever is not possible!")
    # Loading and saving of the retriever is not possible
    json_tricks_loaded = False

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

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
    classes and makes it possible to save the trained models as json file.

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
                retrieval estimator.
        """

        default_parameter = {
            "inputs": {},
            "targets": {},
            "scaler": None,
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

    def is_trained(self):
        """Return true if RetrievalProduct is trained"""
        return self.estimator is not None

    def load_parameters(self, filename):
        """Load the training parameters from a JSON file

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

            # TODO: Change the hard-coded estimator and scaler classes:

            estimator = self._create_model(
                MLPRegressor, estimator["params"], estimator["coefs"],
            )

            # self._set_sklearn_coefs(self.estimator, estimator_coefs)

            scaler = parameter.pop("scaler", None)
            if scaler is None:
                raise ValueError("Found no coefficients for scaler!")

            scaler_class = {
                "standard": StandardScaler,
                "robust": RobustScaler,
                "minmax": MinMaxScaler,
            }

            scaler = self._create_model(
                RobustScaler,  # scaler_class[parameter["scaler_class"]]
                scaler["params"], scaler["coefs"],
            )

            self.estimator = Pipeline([
                ("scaler", scaler,),  ("estimator", estimator)
            ])

            self.parameter.update(parameter)

    def retrieve(self, inputs):
        """Predict the target values for data coming from arrays

        Args:
            inputs: A pandas.DataFrame object. The keys must be the
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
        """ Save the training parameters to a JSON file

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
        #parameter["scaler_class"] = self.scaler_class
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

    def train(self, inputs, targets):
        """Train this retriever with data from arrays

        Args:
            inputs: A pandas.DataFrame with input data.
            targets: A pandas.DataFrame with training data.

        Returns:
            A float number indicating the training score.
        """

        # The input and target labels will be saved because they can be used in
        # other methods as well.
        self.parameter["inputs"] = inputs.columns.tolist()
        self.parameter["targets"] = targets.columns.tolist()

        # Unleash the trainer
        if self.trainer is not None:
            self.trainer.fit(inputs, targets.values.ravel())

            # Use the best estimator from now on:
            self.estimator = self.trainer.best_estimator_
        elif self.estimator is not None:
            self.estimator.fit(inputs, targets.values.ravel())
        else:
            raise ValueError("I need either a trainer or an estimator object!")

        return self.estimator.score(inputs, targets)