import json

import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.neural_network as nn
from sklearn.preprocessing import StandardScaler

from .datasets import Dataset, JointData

__all__ = [
    'Retriever',
]


class Retriever:
    """Still under development.

    """

    def __init__(
            self, parameter=None, parameters_file=None,
            classifier=None, regressor=None,):
        """

        Either a file with previously stored training parameters must given or
        the training parameters must be set by oneself.

        Args:
            parameter: A dictionary with training parameters.
            parameters_file: Name of file with previously stored training
                parameters.
        """

        default_parameter = {
            "inputs": {},
            "targets": {},
            "scaling": None,
        }

        if parameter is None:
            parameter = {}

        self.parameter = default_parameter.update(parameter)

        if parameters_file is not None:
            self.load_parameters(parameters_file)

        # The learners for this RetrievalDataset:
        # self._classifier = None
        self._regressor = None
        self._init_learners(classifier, regressor)

        # The data must be scaled before passed to the regressor. Always the
        # same scaling must be applied.
        self._scaler = None
        if self.parameter["scaling"] is not None:
            self._scaler = StandardScaler()
            self._scaler.scale_ = self.parameter["scaling"].copy()

    def _init_learners(self, classifier, regressor):
        # if classifier is None:
        #     self._classifier = nn.MLPClassifier(
        #         solver='lbfgs',
        #         alpha=1e-5,
        #         hidden_layer_sizes=20,
        #         random_state=1
        #     )
        # else:
        #     self._classifier = classifier

        if regressor is None:
            self._regressor = nn.MLPRegressor(
                solver='lbfgs',
                alpha=1e-5,
                hidden_layer_sizes=20,
                random_state=1
            )
        else:
            self._regressor = regressor

        # Load the coefficients (weights) if they have been already trained:
        # try:
        #     self._classifier.coefs_ = self.parameter["classifier_weights"]
        # except KeyError:
        #     pass

        try:
            self._regressor.coefs_ = self.parameter["regressor_weights"]
        except KeyError:
            pass

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

        try:
            self.parameter["regressor_weights"] = self._regressor.coefs_
        except KeyError:
            pass

        with open(filename, 'w') as outfile:
            json.dump(self.parameter, outfile)

    def run(self, start, end, sources, inputs, output=None):
        """Retrieve the data between two dates

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            sources: Sources where the data come from. Can be one Dataset or
                dict-like object with numpy arrays, or a list / dict of them.
            inputs: A dictionary of input field names. The keys must be the
                same labels as used in :meth:`train`. The values are the field
                names in the original data coming from *sources*.
            output: Either None (default), a path as string containing
                placeholders or a Dataset-like object. If None, all data will
                be returned as one object.

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

        joint_data = JointData(sources)

        for data in joint_data.get(start, end):
            input_data = np.column_stack([
                data[field]
                for _, field in sorted(inputs.items())
            ]).T

            # Scale the input data:
            input_data = self._scaler.transform(input_data)

            # Retrieve the data from the neural network.
            retrieved_data = self._regressor.predict(input_data)
            times = retrieved_data["time"].min(), retrieved_data["time"].max()

            # Store the generated data.
            output.write(retrieved_data, times=times, in_background=True)

    def train(self, start, end, sources, inputs, targets):
        """Train this retriever

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            sources:
            inputs: A dictionary of input field names. The keys are labels of
                the input fields. The values are the field names in the
                original data coming from *sources*.
            targets: A dictionary of target field names. The keys are labels of
                the target fields. The values are the field names in the
                original data coming from *sources*.

        Returns:

        """

        joint_data = JointData(sources)

        for data in joint_data.get(start, end):
            input_data = np.column_stack([
                data[field]
                for _, field in sorted(inputs.items())
            ]).T

            # We have not prepared a scaler yet. This must be fit only once.
            # Normally, it should be fit to all data but we use just the first
            # bin for performance issues.
            if self._scaler is None:
                self._scaler = StandardScaler()
                self._scaler.fit(input_data)

            # Scale the input data:
            input_data = self._scaler.transform(input_data)

            target_data = np.column_stack([
                data[field]
                for _, field in sorted(targets.items())
            ]).T

            train_inputs, test_inputs, train_targets, test_targets = \
                train_test_split(input_data, target_data, test_size=0.2,)

            # Train the neural network
            self._regressor.partial_fit(input_data, target_data)

            print(self._regressor.score(test_inputs, test_targets))
