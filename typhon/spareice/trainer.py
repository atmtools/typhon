#!/usr/bin/env python3

import json

import sklearn.neural_network as nn

__all__ = [
    'Trainer',
    ]

class Trainer:
    """Still under development.

    """

    def __init__(self, dataset, parameter_file=None, inputs=(), outputs=(), targets=(), hidden_layer_sizes=(20,)):
        """
        
        Either a file with previously stored training parameters must given or the training parameters must be set by 
        oneself.
        
        Args:
            dataset (typhon.spareice.TrainerDataset): 
            parameter_file (str): Name of file with previously stored training parameters. 
            inputs (tuple): Field names of the input data.
            outputs (tuple): Field names of the outputs data.
            targets (tuple): Field names of the targets data.
            hidden_layer_sizes (tuple): The ith element is the number of the neurons in the ith hidden layer.
        """

        self.dataset = dataset

        self.hidden_layer_sizes = hidden_layer_sizes

        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets

        # The neural networks for this trainer.
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

        ...

    def save_parameters(self, filename):
        """ Saves the training parameters as a JSON file.
        
        Training parameters are:
            - weights of the neural networks (classifier and regressor)
            - names of the input, output, and target fields
        
        Args:
            filename (str): The name of file where to store the training parameters. If it does not end with ".json", 
                the suffix will be added to the filename.

        Returns:
            None
        """

        if not filename.endswith(".json"):
            filename += ".json"

        # Create the dictionary of training parameters.
        parameter = {
            "hidden_layer_sizes" : self.hidden_layer_sizes,
            "inputs" : self.inputs,
            "outputs" : self.outputs,
            "targets" : self.targets
        }

        try:
            parameter["classifier_weights"] = self.classifier.coefs_
        except:
            print("WARNING: The classifier NN has not been trained yet.")

        try:
            parameter["regressor_weights"] = self.regressor.coefs_
        except:
            print("WARNING: The regressor NN has not been trained yet.")

        with open(filename, 'w') as outfile:
            json.dump(parameter, outfile)

    def train(self, episodes, batch_training=True):
        """
        
        Args:
            episodes: 
            batch_training: 

        Returns:
            
        """

        for file, timestamp in self.dataset.find_files():
            data = self.dataset.read_file(file)

            # Generate the training tuples.
            # CAUTION: The data should be sorted equally!
            input_data = zip(*[data[input_var] for input_var in self.inputs])
            target_data = zip(*[data[target_var] for target_var in self.targets])

            # Train the neural network
            self.regressor.fit(input_data, target_data)