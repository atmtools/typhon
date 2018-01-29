#!/usr/bin/env python3

import json

import sklearn.neural_network as nn

__all__ = [
    'Retriever',
    ]


class Retriever:
    """Still under development.

    """
    def __init__(self, inputs, output, parameters_file):
        """
        
        Args:
            inputs: A list of data sources where the input fields should come
                from. A data source can either be a dict-like array set or a
                Dataset which read method returns such an array set.
            output: The dataset where it should store the retrieved data.
            parameters_file: The name of the file with the parameters
                that were used for training SPARE-ICE and  now will be used for
                retrieving. Should be created with the
                :meth:`~typhon.spareice.Trainer.save_parameters` method first.
        """

        self.inputs = inputs
        self.output = output

        # Declare all other member variables which will be defined by
        # self.load_parameters().
        self.hidden_layer_sizes = None

        self.inputs = None
        self.outputs = None
        self.targets = None

        # The neural networks for this trainer.
        self.classifier = None
        self.regressor = None

        # Load from parameters file
        self.load_parameters(parameters_file)

    def load_parameters(self, filename):
        """ Loads the training parameters from a JSON file.

        Training parameters are:
            - weights of the neural networks (classifier and regressor)
            - names of the input, output, and target fields

        Args:
            filename (str): The name of file from where to load the training parameters.

        Returns:
            None
        """

        with open(filename, 'r') as infile:
            parameter = json.load(infile)

            self.hidden_layer_sizes = parameter["hidden_layer_sizes"]

            self.inputs = parameter["inputs"]
            self.outputs = parameter["outputs"]
            self.targets = parameter["targets"]

            self.classifier = nn.MLPClassifier(
                solver='lbfgs',
                alpha=1e-5,
                hidden_layer_sizes=self.hidden_layer_sizes,
                random_state=1
            )
            self.regressor = nn.MLPRegressor(
                solver='lbfgs',
                alpha=1e-5,
                hidden_layer_sizes=self.hidden_layer_sizes,
                random_state=1
            )

            try:
                self.classifier.coefs_ = parameter["classifier_weights"]
            except KeyError:
                print("WARNING: The classifier NN has not been trained yet.")

            try:
                self.regressor.coefs_ = parameter["regressor_weights"]
            except KeyError:
                print("WARNING: The regressor NN has not been trained yet.")

    def retrieve(self, start=None, end=None):
        """
        
        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".

        Returns:
            None
        """

        for file in self.inputs.find_files(start, end):
            data = self.input_dataset.read_file(file)

            # Generate the training tuples.
            # CAUTION: The data should be sorted equally!
            input_data = zip(*[data[input_var] for input_var in self.inputs])

            # Retrieve the data from the neural network.
            spareice_data = self.regressor.predict(input_data)

            # Store the generated data.
            self.output_dataset.write_file(timestamp, spareice_data)
