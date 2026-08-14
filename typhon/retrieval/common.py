from ast import literal_eval
import copy
from importlib import import_module
import inspect

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

__all__ = [
    'RetrievalProduct',
]

# Scikit-learn <1.0 used un-prefixed module names (e.g. sklearn.tree.tree).
# Models saved with such versions must be remapped to the current module
# names when they are deserialized:
_SKLEARN_MODULE_MAP = {
    'sklearn.preprocessing.data':
        'sklearn.preprocessing._data',
    'sklearn.neural_network.multilayer_perceptron':
        'sklearn.neural_network._multilayer_perceptron',
    'sklearn.tree.tree': 'sklearn.tree._classes',
}


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
        module_name = _SKLEARN_MODULE_MAP.get(module_name, module_name)
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
            dtype = item["__dtype__"]
            if isinstance(dtype, str) and dtype.startswith("{"):
                # A structured dtype in its dictionary representation, as
                # produced by str(dtype) of an aligned dtype (numpy >= 2):
                dtype = literal_eval(dtype)
            try:
                return np.array(
                    item["__ndarray__"],
                    dtype=dtype,
                )
            except TypeError:
                if isinstance(dtype, str):
                    dtype = literal_eval(dtype)
                    return np.array(
                        item["__ndarray__"],
                        dtype=dtype,
                    )
                raise

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
        # Newer sklearn versions store the feature count as n_features_in_
        # which may be None for unfitted attributes. Old versions use
        # n_features_ instead.
        n_features = coefs.get("n_features_in_")
        if n_features is None:
            n_features = coefs["n_features_"]
        tree = instance(
            n_features,
            np.atleast_1d(np.asarray(coefs["n_classes_"], dtype=np.intp)),
            coefs["n_outputs_"],
        )

        state = dictionary["coefs"]
        nodes = state.get("nodes")
        if nodes is not None and "missing_go_to_left" not in nodes.dtype.names:
            new_dtype = np.dtype(
                [*nodes.dtype.descr, ("missing_go_to_left", "u1")],
                align=True,
            )
            new_nodes = np.zeros(nodes.shape, dtype=new_dtype)
            for name in nodes.dtype.names:
                new_nodes[name] = nodes[name]
            state["nodes"] = new_nodes

        tree.__setstate__(state)
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
                if not attr.startswith("_repr_")
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
        params = dictionary["params"]
        parameters = inspect.signature(instance.__init__).parameters
        params = {
            key: value for key, value in params.items()
            if key in parameters
        }
        model = instance(**params)
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
