from ast import literal_eval
import copy
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from typhon.utils import to_array

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

    To save this object to a json file, the additional package json_tricks is
    required.
    """

    def __init__(self, verbose=False):
        """Initialize a Retriever object

        Args:
            verbose: The higher this value is the more debug messages are
                printed. Default is False.
        """

        # The trainer and/or model for this retriever:
        self.estimator = None
        self.verbose = verbose
        self._inputs = []
        self._outputs = []

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @staticmethod
    def _import_class(module_name, class_name):
        """Import a class dynamically to the namespace"""
        mod = import_module(module_name)
        klass = getattr(mod, class_name)
        return klass

    @staticmethod
    def _encode_numpy(obj):
        def _to_dict(item):
            if isinstance(item, np.ndarray):
                return {
                    "__ndarray__": item.tolist(),
                    "__dtype__": str(item.dtype),
                    "__shape__": item.shape,
                }
            else:
                return item.item()

        def _is_numpy(item):
            return type(item).__module__ == np.__name__

        if isinstance(obj, dict):
            obj = obj.copy()
            iterator = obj.items()
        elif isinstance(obj, list):
            obj = obj.copy()
            iterator = enumerate(obj)
        else:
            return obj

        for key, value in iterator:
            if _is_numpy(value):
                obj[key] = _to_dict(value)
            elif isinstance(value, (list, dict)):
                obj[key] = RetrievalProduct._encode_numpy(value)

        return obj

    @staticmethod
    def _decode_numpy(obj):
        def _from_dict(item):
            try:
                return np.array(
                    item["__ndarray__"],
                    dtype=item["__dtype__"],
                )
            except TypeError:
                return np.array(
                    item["__ndarray__"],
                    dtype=literal_eval(item["__dtype__"]),
                )

        def _is_numpy(item):
            return isinstance(item, dict) and "__ndarray__" in item

        if isinstance(obj, dict):
            obj = obj.copy()
            iterator = obj.items()
        elif isinstance(obj, list):
            obj = obj.copy()
            iterator = enumerate(obj)
        else:
            return obj

        for key, value in iterator:
            if _is_numpy(value):
                obj[key] = _from_dict(value)
            elif isinstance(value, (list, tuple, dict)):
                obj[key] = RetrievalProduct._decode_numpy(value)

        return obj

    @staticmethod
    def _tree_to_dict(tree):
        return {
            "module": type(tree).__module__,
            "class": type(tree).__name__,
            "coefs": tree.__getstate__(),
        }

    @staticmethod
    def _tree_from_dict(dictionary, coefs):
        instance = RetrievalProduct._import_class(
            dictionary["module"], dictionary["class"]
        )
        tree = instance(
            to_array(coefs["n_features_"]),
            to_array(coefs["n_classes_"]),
            to_array(coefs["n_outputs_"])
        )
        tree.__setstate__(dictionary["coefs"])
        return tree

    @staticmethod
    def _model_to_dict(model):
        """Convert a sklearn model object to a dictionary"""
        dictionary = {
            "module": type(model).__module__,
            "class": type(model).__name__,
            "params": model.get_params(deep=True),
            "coefs": {
                attr: copy.deepcopy(getattr(model, attr))
                for attr in model.__dir__()
                if not attr.startswith("__") and attr.endswith("_")
            }
        }

        if "tree_" in dictionary["coefs"]:
            # Not funny. sklearn.tree objects are not directly
            # serializable to json. Hence, we must dump them by ourselves.
            dictionary["coefs"]["tree_"] = RetrievalProduct._tree_to_dict(
                dictionary["coefs"]["tree_"]
            )

        return RetrievalProduct._encode_numpy(dictionary)

    @staticmethod
    def _model_from_dict(dictionary):
        """Create a sklearn model object from a dictionary"""
        dictionary = RetrievalProduct._decode_numpy(dictionary)
        instance = RetrievalProduct._import_class(
            dictionary["module"], dictionary["class"]
        )
        model = instance(**dictionary["params"])
        for attr, value in dictionary["coefs"].items():
            if attr == "tree_":
                # We must treat a tree specially:
                value = RetrievalProduct._tree_from_dict(
                    value, dictionary["coefs"]
                )
            try:
                setattr(model, attr, value)
            except AttributeError:
                # Some attributes cannot be set such as feature_importances_
                pass
        return model

    @staticmethod
    def _pipeline_to_dict(pipeline):
        """Convert a pipeline object to a dictionary"""
        if pipeline is None:
            raise ValueError("No object trained!")

        all_steps = {}
        for name, model in pipeline.steps:
            all_steps[name] = RetrievalProduct._model_to_dict(model)
        return all_steps

    @staticmethod
    def _pipeline_from_dict(dictionary):
        """Create a pipeline object from a dictionary"""
        all_steps = []
        for name, step in dictionary.items():
            model = RetrievalProduct._model_from_dict(step)
            all_steps.append([name, model])

        return Pipeline(all_steps)

    def is_trained(self):
        """Return true if RetrievalProduct is trained"""
        return self.estimator is not None

    @classmethod
    def from_dict(cls, parameter, *args, **kwargs):
        """Load a retrieval product from a dictionary

        Args:
            parameter: A dictionary with the training parameters. Simply the
                output of :meth:`to_dict`.
            *args: Positional arguments allowed for :meth:`__init__`.
            **kwargs Keyword arguments allowed for :meth:`__init__`.

        Returns:
            A new :class:`RetrievalProduct` object.
        """

        self = cls(*args, **kwargs)

        estimator = parameter.get("estimator", None)
        if estimator is None:
            raise ValueError("Found no coefficients for estimator!")

        is_pipeline = parameter["estimator_is_pipeline"]

        if is_pipeline:
            self.estimator = self._pipeline_from_dict(estimator)
        else:
            self.estimator = self._model_from_dict(estimator)

        self._inputs = parameter["inputs"]
        self._outputs = parameter["outputs"]
        return self

    def to_dict(self):
        """Dump this retrieval product to a dictionary"""
        parameter = {}
        if isinstance(self.estimator, Pipeline):
            parameter["estimator"] = self._pipeline_to_dict(self.estimator)
            parameter["estimator_is_pipeline"] = True
        else:
            parameter["estimator"] = self._model_to_dict(self.estimator)
            parameter["estimator_is_pipeline"] = False

        parameter["inputs"] = self.inputs
        parameter["outputs"] = self.outputs
        return parameter

    @classmethod
    def from_txt(cls, filename, *args, **kwargs):
        """Load a retrieval product from a txt file

        Notes:
            The output format is not standard json!

        Training parameters are:
        * weights of the estimator
        * names of the input and target fields

        Args:
            filename: The name of file from where to load the training
                parameters.
            *args: Positional arguments allowed for :meth:`__init__`.
            **kwargs Keyword arguments allowed for :meth:`__init__`.

        Returns:
            A new :class:`RetrievalProduct` object.
        """

        with open(filename, 'r') as infile:
            parameter = literal_eval(infile.read())
            return cls.from_dict(parameter, *args, **kwargs)

    def to_txt(self, filename):
        """Save this retrieval product to a txt file

        Training parameters are:
        * configuration of the used estimator
        * names of the input, output, and target fields

        Args:
            filename: The name of the file where to store the training
                parameters.

        Returns:
            None
        """

        with open(filename, 'w') as outfile:
            outfile.write(repr(self.to_dict()))

    def retrieve(self, inputs):
        """Predict the target values for data coming from arrays

        Args:
            inputs: A pandas.DataFrame object. The keys must be the
                same labels as used in :meth:`train`.

        Returns:
             A pandas.DataFrame object with the retrieved data.

        Examples:

        .. :code-block:: python

            # TODO
        """

        if self.estimator is None:
            raise NotTrainedError()

        # Skip empty datasets
        if inputs.empty:
            return None

        # Retrieve the data from the neural network:
        output_data = self.estimator.predict(inputs)

        return pd.DataFrame(data=output_data, columns=self.outputs)

    def score(self, inputs, targets):
        """

        Args:
            inputs: A pandas.DataFrame with input data.
            targets: A pandas.DataFrame with target data.

        Returns:
            The metric score as a number
        """
        if self.estimator is None:
            raise NotTrainedError()

        return self.estimator.score(inputs.squeeze(), targets.squeeze())

    def train(self, estimator, inputs, targets):
        """Train this retriever with data from arrays

        Args:
            estimator: The object that will be trained. If it is a trainer
                object such as a GridSearchCV, the best estimator will be
                chosen after training. Can also be a Pipeline or a standard
                Estimator from scikit-learn.
            inputs: A pandas.DataFrame with input data.
            targets: A pandas.DataFrame with target data.

        Returns:
            A float number indicating the training score.
        """

        # The input and target labels will be saved because to know what this
        # product retrieves and from what:
        self._inputs = inputs.columns.tolist()
        self._outputs = targets.columns.tolist()

        # Start to train!
        estimator.fit(inputs.squeeze(), targets.squeeze())

        # Let's check whether the estimator was a trainer object such as
        # GridSearchCV, etc. Then we save only the best estimator.
        if hasattr(estimator, "best_estimator_"):
            # Use the best estimator from now on:
            self.estimator = estimator.best_estimator_
        else:
            self.estimator = estimator

        return self.score(inputs, targets)
