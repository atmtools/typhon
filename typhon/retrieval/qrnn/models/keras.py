import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K
import logging

################################################################################
# Quantile loss
################################################################################

logger = logging.getLogger(__name__)

def skewed_absolute_error(y_true, y_pred, tau):
    """
    The quantile loss function for a given quantile tau:

    L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)

    Where I is the indicator function.
    """
    dy = y_pred - y_true
    return K.mean((1.0 - tau) * K.relu(dy) + tau * K.relu(-dy), axis=-1)


def quantile_loss(y_true, y_pred, taus):
    """
    The quantiles loss for a list of quantiles. Sums up the error contribution
    from the each of the quantile loss functions.
    """
    e = skewed_absolute_error(
        K.flatten(y_true), K.flatten(y_pred[:, 0]), taus[0])
    for i, tau in enumerate(taus[1:]):
        e += skewed_absolute_error(K.flatten(y_true),
                                   K.flatten(y_pred[:, i + 1]),
                                   tau)
    return e


class QuantileLoss:
    """
    Wrapper class for the quantile error loss function. A class is used here
    to allow the implementation of a custom `__repr` function, so that the
    loss function object can be easily loaded using `keras.model.load`.

    Attributes:

        quantiles: List of quantiles that should be estimated with
                   this loss function.

    """

    def __init__(self, quantiles):
        self.__name__ = "QuantileLoss"
        self.quantiles = quantiles

    def __call__(self, y_true, y_pred):
        return quantile_loss(y_true, y_pred, self.quantiles)

    def __repr__(self):
        return "QuantileLoss(" + repr(self.quantiles) + ")"


class KerasModel:
    def __init__(self):
        pass

    def train(self):
        pass


class FullyConnected(Sequential):
    """
    Convenience class to represent fully connected models with
    given number of input and output features and depth and
    width of hidden layers.
    """
    def __init__(self,
                 input_dimension,
                 quantiles,
                 arch,
                 activation = "relu"):
        """
        Create a fully-connected neural network.

        Args:
            input_dimension(int): Number of input features
            output_dimension(int): Number of output features
            arch(tuple): Tuple (d, w) of d, the number of hidden
                layers in the network, and w, the width of the net-
                work.
            activation(str or None): Activation function to insert between
                hidden layers. The given string is passed to the
                keras.layers.Activation class. If None no activation function
                is used.
        """
        quantiles = np.array(quantiles)
        output_dimension = quantiles.size
        if len(arch) == 0:
            layers = [Dense(output_dimension, input_shape=(input_dimension))]
        else:
            d, w = arch
            layers = [Dense(input_dimension, input_shape=(w,))]
            for i in range(d - 1):
                layers.append(Dense(w, input_shape=(w,)))
                if not activation is None:
                    layers.append(Activation(activation))
            layers.append(Dense(output_dimension, input_shape=(w,)))
        super().__init__(layers)
