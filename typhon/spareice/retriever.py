#!/usr/bin/env python3

import json

import sklearn.neural_network as nn

__all__ = [
    'Retriever',
    ]

class Retriever:
    """

    """
    def __init__(self, input_dataset, output_dataset, parameters_file):
        """
        
        Args:
            input_dataset (typhon.datasets.Dataset): The dataset where it should retrieve from.
            output_dataset (typhon.datasets.Dataset): The dataset where it should store the retrieved data.
            parameters_file (str): The name of the file with the parameters that were used for training SPARE-ICE and 
                now will be used for rerieving. Should be created with the typhon.spareice.Trainer.save_parameters() 
                method first.
        """

        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        ## Declare all other member variables which will be defined by self.load_parameters().
        self.hidden_layer_sizes = None

        self.inputs = None
        self.outputs = None
        self.targets = None

        # The neural networks for this trainer.
        self.classifier = None
        self.regressor = None

        ## Load from parameters file
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



    def retrieve(self, start=(0, 0, 0), end=(9999, 12, 31)):
        """
        
        Args:
            start (tuple): Starting date from when to retrieve.
            end (tuple): Ending date to when to retrieve.

        Returns:
            None
        """

        for file, timestamp in self.input_dataset.find_files(all=True):
            data = self.input_dataset.read_file(file)

            # Generate the training tuples.
            # CAUTION: The data should be sorted equally!
            input_data = zip(*[data[input_var] for input_var in self.inputs])

            # Retrieve the data from the neural network.
            spareice_data = self.regressor.predict(input_data)

            # Store the generated data.
            self.output_dataset.write_file(timestamp, spareice_data)
