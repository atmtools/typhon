"""
typhon.retrieval.qrnn.models.pytorch.fully_connected
====================================================

This module provides an implementation of a fully-connected feed forward
neural network in pytorch.
"""
from torch import nn
from typhon.retrieval.qrnn.models.pytorch.common import PytorchModel, activations

################################################################################
# Fully-connected network
################################################################################


class FullyConnected(PytorchModel, nn.Sequential):
    """
    Pytorch implementation of a fully-connected QRNN model.
    """

    def __init__(self, input_dimension, quantiles, arch):

        """
        Create a fully-connected neural network.

        Args:
            input_dimension(:code:`int`): Number of input features
            quantiles(:code:`array`): The quantiles to predict given
                as fractions within [0, 1].
            arch(tuple): Tuple :code:`(d, w, a)` containing :code:`d`, the
                number of hidden layers in the network, :code:`w`, the width
                of the network and :code:`a`, the type of activation functions
                to be used as string.
        """
        PytorchModel.__init__(self, input_dimension, quantiles)
        output_dimension = quantiles.size
        self.arch = arch

        if len(arch) == 0:
            layers = [nn.Linear(input_dimension, output_dimension)]
        else:
            d, w, act = arch
            if isinstance(act, str):
                act = activations[act]
            layers = [nn.Linear(input_dimension, w)]
            for _ in range(d - 1):
                layers.append(nn.Linear(w, w))
                if act is not None:
                    layers.append(act())
            layers.append(nn.Linear(w, output_dimension))
        nn.Sequential.__init__(self, *layers)
