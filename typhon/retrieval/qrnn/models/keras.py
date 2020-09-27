"""
typhon.retrieval.qrnn.models.keras
==================================

This module provides Keras neural network models that can be used as backend
models with the :py:class:`typhon.retrieval.qrnn.QRNN` class.
"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, deserialize
from keras.optimizers import SGD
import keras.backend as K
import logging


def save_model(f, model):
    """
    Save keras model.

    Args:
        f(:code:`str` or binary stream): Either a path or a binary stream
            to store the data to.
        model(:code:`keras.models.Models`): The Keras model to save
    """
    keras.models.save_model(model, f)


def load_model(f, quantiles):
    """
    Load keras model.

    Args:
        f(:code:`str` or binary stream): Either a path or a binary stream
            to read the model from
        quantiles(:code:`np.ndarray`): Array containing the quantiles
            that the model predicts.

    Returns:
        The loaded keras model.
    """
    #
    # This is a bit hacky but seems required to handle
    # the custom model classes.
    #
    def make_fully_connected(layers=None, **kwargs):
        layers = list(map(deserialize, layers))
        input_dimensions = layers[0].batch_input_shape[1]
        return FullyConnected(input_dimensions, quantiles, (), layers)

    custom_objects = {
        "FullyConnected": make_fully_connected,
        "QuantileLoss": QuantileLoss,
    }
    model = keras.models.load_model(f, custom_objects=custom_objects)
    return model


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
    e = skewed_absolute_error(K.flatten(y_true), K.flatten(y_pred[:, 0]), taus[0])
    for i, tau in enumerate(taus[1:]):
        e += skewed_absolute_error(K.flatten(y_true), K.flatten(y_pred[:, i + 1]), tau)
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


################################################################################
# Keras data generators
################################################################################


class BatchedDataset:
    """
    Keras data loader that batches a given dataset of numpy arryas.
    """

    def __init__(self, training_data, batch_size):
        """
        Create batched dataset.

        Args:
            training_data: Tuple :code:`(x, y)` containing the input
               and output data as arrays.
            batch_size(:code:`int`): The batch size
        """
        x, y = training_data
        self.x = x
        self.y = y
        self.bs = batch_size
        self.indices = np.random.permutation(x.shape[0])
        self.i = 0

    def __iter__(self):
        logger.info("iter...")
        return self

    def __len__(self):
        return self.x.shape[0] // self.bs

    def __next__(self):
        inds = self.indices[
            np.arange(self.i * self.bs, (self.i + 1) * self.bs) % self.indices.size
        ]
        x_batch = np.copy(self.x[inds, :])
        y_batch = self.y[inds]
        self.i = self.i + 1
        # Shuffle training set after each epoch.
        if self.i % (self.x.shape[0] // self.bs) == 0:
            self.indices = np.random.permutation(self.x.shape[0])

        return (x_batch, y_batch)


class TrainingGenerator:
    """
    This Keras sample generator takes a generator for noise-free training data
    and adds independent Gaussian noise to each of the components of the input.

    Attributes:
        training_data: Data generator providing the data
        sigma_noise: A vector containing the standard deviation of each
                 component.
    """

    def __init__(self, training_data, sigma_noise=None):
        """
        Args:
            training_data: Data generator providing the original (noise-free)
                training data.
            sigma_noise: Vector the length of the input dimensions specifying
                the standard deviation of the noise.
        """
        self.training_data = training_data
        self.sigma_noise = sigma_noise

    def __iter__(self):
        logger.info("iter...")
        return self

    def __len__(self):
        return len(self.training_data)

    def __next__(self):
        x_batch, y_batch = next(self.training_data)
        if not self.sigma_noise is None:
            x_batch += np.random.randn(*x_batch.shape) * self.sigma_noise
        return (x_batch, y_batch)


class AdversarialTrainingGenerator:
    """
    This Keras sample generator takes the noise-free training data
    and adds independent Gaussian noise to each of the components
    of the input.

    Attributes:
        training_data: Training generator to use to generate the input
            data
        input_gradients: Keras function to compute the gradients of the
            network
        eps: The perturbation factor.
    """

    def __init__(self, training_data, input_gradients, eps):
        """
        Args:
            training_data: Training generator to use to generate the input
                data
            input_gradients: Keras function to compute the gradients of the
                network
            eps: The perturbation factor.
        """
        self.training_data = training_data
        self.input_gradients = input_gradients
        self.eps = eps

    def __iter__(self):
        logger.info("iter...")
        return self

    def __len__(self):
        return len(self.training_data)

    def __next__(self):
        if self.i % 2 == 0:
            x_batch, y_batch = next(self.training_data)
            self.x_batch = x_batch
            self.y_batch = y_batch
        else:
            x_batch = self.x_batch
            y_batch = self.y_batch
            grads = self.input_gradients([x_batch, y_batch, 1.0])
            x_batch += self.eps * np.sign(grads)

        self.i = self.i + 1
        return x_batch, y_batch


class ValidationGenerator:
    """
    This Keras sample generator is similar to the training generator
    only that it returns the whole validation set and doesn't perform
    any randomization.

    Attributes:

        x_val: The validation input, i.e. the brightness temperatures
                 measured by the satellite.
        y_val: The validation output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
    """

    def __init__(self, validation_data, sigma_noise):
        self.validation_data = validation_data
        self.sigma_noise = sigma_noise

    def __iter__(self):
        return self

    def __next__(self):
        x_val, y_val = next(self.validation_data)
        if not self.sigma_noise is None:
            x_val += np.random.randn(*self.x_val.shape) * self.sigma_noise
        return (x_val, self.y_val)


################################################################################
# LRDecay
################################################################################


class LRDecay(keras.callbacks.Callback):
    """
    The LRDecay class implements the Keras callback interface and reduces
    the learning rate according to validation loss reduction.

    Attributes:

        lr_decay: The factor c > 1.0 by which the learning rate is
                  reduced.
        lr_minimum: The training is stopped when this learning rate
                    is reached.
        convergence_steps: The number of epochs without validation loss
                           reduction required to reduce the learning rate.

    """

    def __init__(self, model, lr_decay, lr_minimum, convergence_steps):
        self.model = model
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.convergence_steps = convergence_steps
        self.steps = 0

    def on_train_begin(self, logs={}):
        self.losses = []
        self.steps = 0
        self.min_loss = 1e30

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get("val_loss")
        if loss is None:
            loss = logs.get("loss")
        self.losses += [loss]
        if not self.losses[-1] < self.min_loss:
            self.steps = self.steps + 1
        else:
            self.steps = 0
        if self.steps > self.convergence_steps:
            lr = keras.backend.get_value(self.model.optimizer.lr)
            keras.backend.set_value(self.model.optimizer.lr, lr / self.lr_decay)
            self.steps = 0
            logger.info("\n Reduced learning rate to " + str(lr))

            if lr < self.lr_minimum:
                self.model.stop_training = True

        self.min_loss = min(self.min_loss, self.losses[-1])


################################################################################
# QRNN
################################################################################


class KerasModel:
    r"""
    Base class for Keras models.

    This base class provides generic utility function for the training, saving
    and evaluation of Keras models.

    Attributes:
        input_dimensions (int): The input dimension of the neural network, i.e.
            the dimension of the measurement vector.
        quantiles (numpy.array): The 1D-array containing the quantiles
            :math:`\tau \in [0, 1]` that the network learns to predict.

        depth (int):
            The number layers in the network excluding the input layer.

        width (int):
            The width of the hidden layers in the network.

        activation (str):
            The name of the activation functions to use in the hidden layers
            of the network.

        models (list of keras.models.Sequential):
            The ensemble of Keras neural networks used for the quantile regression
            neural network.
    """

    def __init__(self, input_dimension, quantiles):
        """
        Create a QRNN model.

        Arguments:
            input_dimension(int): The dimension of the measurement space, i.e. the number
                            of elements in a single measurement vector y

            quantiles(np.array): 1D-array containing the quantiles  to estimate of
                                 the posterior distribution. Given as fractions
                                 within the range [0, 1].
        """
        self.input_dimension = input_dimension
        self.quantiles = np.array(quantiles)

    def reset(self):
        """
        Reinitialize the state of the model.
        """
        self.reset_states()

    def train(
        self,
        training_data,
        validation_data=None,
        batch_size=256,
        sigma_noise=None,
        adversarial_training=False,
        delta_at=0.01,
        initial_learning_rate=1e-2,
        momentum=0.0,
        convergence_epochs=5,
        learning_rate_decay=2.0,
        learning_rate_minimum=1e-6,
        maximum_epochs=200,
        training_split=0.9,
        gpu=False,
    ):

        if type(training_data) == tuple:
            if not type(training_data[0]) == np.ndarray:
                raise ValueError(
                    "When training data is provided as tuple"
                    " (x, y) it must contain numpy arrays."
                )
            training_data = BatchedDataset(training_data, batch_size)

        if type(validation_data) is tuple:
            validation_data = BatchedDataset(validation_data, batch_size)

        loss = QuantileLoss(self.quantiles)

        # Compile model
        self.custom_objects = {loss.__name__: loss}
        optimizer = SGD(lr=initial_learning_rate)
        self.compile(loss=loss, optimizer=optimizer)

        #
        # Setup training generator
        #
        training_generator = TrainingGenerator(training_data, sigma_noise)
        if adversarial_training:
            inputs = [self.input, self.targets[0], self.sample_weights[0]]
            input_gradients = K.function(
                inputs, K.gradients(self.total_loss, self.input)
            )
            training_generator = AdversarialTrainingGenerator(
                training_generator, input_gradients, delta_at
            )

        if validation_data is None:
            validation_generator = None
        else:
            validation_generator = ValidationGenerator(validation_data, sigma_noise)
        lr_callback = LRDecay(
            self, learning_rate_decay, learning_rate_minimum, convergence_epochs
        )
        self.fit_generator(
            training_generator,
            steps_per_epoch=len(training_generator),
            epochs=maximum_epochs,
            validation_data=validation_generator,
            validation_steps=1,
            callbacks=[lr_callback],
        )


################################################################################
# Fully-connected network
################################################################################


class FullyConnected(KerasModel, Sequential):
    """
    Keras implementation of fully-connected networks.
    """

    def __init__(self, input_dimension, quantiles, arch, layers=None):
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
        quantiles = np.array(quantiles)
        output_dimension = quantiles.size

        if layers is None:
            if len(arch) == 0:
                layers = [Dense(output_dimension, input_shape=(input_dimension))]
            else:
                d, w, a = arch
                layers = [Dense(w, input_shape=(input_dimension,))]
                for i in range(d - 1):
                    layers.append(Dense(w, input_shape=(w,)))
                    if not a is None:
                        layers.append(Activation(a))
                layers.append(Dense(output_dimension, input_shape=(w,)))

        KerasModel.__init__(self, input_dimension, quantiles)
        Sequential.__init__(self, layers)
