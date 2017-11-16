import numpy as np
import matplotlib
import copy
import os
import pickle

# Keras Imports
try:
    import keras
    from keras.models import Sequential, clone_model
    from keras.layers import Dense, Activation, Dropout
    from keras.optimizers import SGD
except ImportError:
    raise Exception("Could not import the required Keras modules. The QRNN " /
                    "implementation was developed for use with Keras version" /
                    "2.0.9.")

################################################################################
# Loss Functions
################################################################################

import keras.backend as K


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
        self.__name__ = "Quantile Loss"
        self.quantiles = quantiles

    def __call__(self, y_true, y_pred):
        return quantile_loss(y_true, y_pred, self.quantiles)

    def __repr__(self):
        return "QuantileLoss(" + repr(self.quantiles) + ")"

################################################################################
# Keras Interface Classes
################################################################################


class TrainingGenerator:
    """
    This Keras sample generator takes the noise-free training data
    and adds independent Gaussian noise to each of the components
    of the input.

    Attributes:

        x_train: The training input, i.e. the brightness temperatures
                 measured by the satellite.
        y_train: The training output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
        batch_size: The size of a training batch.
    """

    def __init__(self, x_train, x_mean, x_sigma, y_train, sigma_noise, batch_size):
        self.bs = batch_size

        self.x_train = x_train
        self.x_mean = x_mean
        self.x_sigma = x_sigma
        self.y_train = y_train
        self.sigma_noise = sigma_noise

        self.indices = np.random.permutation(x_train.shape[0])
        self.i = 0

    def __iter__(self):
        print("iter...")
        return self

    def __next__(self):
        inds = self.indices[np.arange(self.i * self.bs,
                                      (self.i + 1) * self.bs)
                            % self.indices.size]
        x_batch = np.copy(self.x_train[inds, :])
        if not self.sigma_noise is None:
            x_batch += np.random.randn(*x_batch.shape) * self.sigma_noise
        x_batch = (x_batch - self.x_mean) / self.x_sigma
        y_batch = self.y_train[inds]

        self.i = self.i + 1

        # Shuffle training set after each epoch.
        if self.i % (self.x_train.shape[0] // self.bs) == 0:
            self.indices = np.random.permutation(self.x_train.shape[0])

        return (x_batch, y_batch)

# TODO: Make y-noise argument optional


class AdversarialTrainingGenerator:
    """
    This Keras sample generator takes the noise-free training data
    and adds independent Gaussian noise to each of the components
    of the input.

    Attributes:

        x_train: The training input, i.e. the brightness temperatures
                 measured by the satellite.
        y_train: The training output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
        batch_size: The size of a training batch.
    """

    def __init__(self,
                 x_train,
                 x_mean,
                 x_sigma,
                 y_train,
                 sigma_noise,
                 batch_size,
                 input_gradients,
                 eps):
        self.bs = batch_size

        self.x_train = x_train
        self.x_mean = x_mean
        self.x_sigma = x_sigma
        self.y_train = y_train
        self.sigma_noise = sigma_noise

        self.indices = np.random.permutation(x_train.shape[0])
        self.i = 0

        # compile gradient function
        bs2 = self.bs // 2

        self.input_gradients = input_gradients
        self.eps = eps

    def __iter__(self):
        print("iter...")
        return self

    def __next__(self):

        if self.i == 0:
            inds = np.random.randint(0, self.x_train.shape[0], self.bs)

            x_batch = np.copy(self.x_train[inds, :])
            if (self.sigma_noise):
                x_batch += np.random.randn(*x_batch.shape) * self.sigma_noise

            x_batch = (x_batch - self.x_mean) / self.x_sigma
            y_batch = self.y_train[inds]

        else:

            bs2 = self.bs // 2
            inds = np.random.randint(0, self.x_train.shape[0], bs2)

            x_batch = np.zeros((self.bs, self.x_train.shape[1]))
            y_batch = np.zeros((self.bs, 1))

            x_batch[:bs2, :] = np.copy(self.x_train[inds, :])
            if (self.sigma_noise):
                x_batch[:bs2, :] += np.random.randn(bs2, self.x_train.shape[1]) \
                    * self.sigma_noise
            x_batch[:bs2, :] = (x_batch[:bs2, :] - self.x_mean) / self.x_sigma
            y_batch[:bs2, :] = self.y_train[inds].reshape(-1, 1)
            x_batch[bs2:, :] = x_batch[:bs2, :]
            y_batch[bs2:, :] = y_batch[:bs2, :]

            if (self.i > 10):
                grads = self.input_gradients(
                    [x_batch[:bs2, :], y_batch[:bs2, :], [1.0]])[0]
                x_batch[bs2:, :] += self.eps * np.sign(grads)

        self.i = self.i + 1
        return (x_batch, y_batch)


# TODO: Make y-noise argument optional
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

    def __init__(self, x_val, x_mean, x_sigma, y_val, sigma_noise):
        self.x_val = x_val
        self.x_mean = x_mean
        self.x_sigma = x_sigma

        self.y_val = y_val

        self.sigma_noise = sigma_noise

    def __iter__(self):
        return self

    def __next__(self):
        x_val = np.copy(self.x_val)
        if not self.sigma_noise is None:
            x_val += np.random.randn(*self.x_val.shape) * self.sigma_noise
        x_val = (x_val - self.x_mean) / self.x_sigma
        return (x_val, self.y_val)


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
        self.losses += [logs.get('val_loss')]
        if not self.losses[-1] < self.min_loss:
            self.steps = self.steps + 1
        else:
            self.steps = 0
        if self.steps > self.convergence_steps:
            lr = keras.backend.get_value(self.model.optimizer.lr)
            keras.backend.set_value(
                self.model.optimizer.lr, lr / self.lr_decay)
            self.steps = 0
            print("\n Reduced learning rate to " + str(lr))

            if lr < self.lr_minimum:
                self.model.stop_training = True

        self.min_loss = min(self.min_loss, self.losses[-1])

################################################################################
# QRNN
################################################################################


class QRNN:
    r"""
    Quantile Regression Neural Network (QRNN)

    This class implements quantile regression neural networks and can be used
    to estimate quantiles of the posterior distribution of remote sensing
    retrievals.

    Internally, the QRNN uses a feed-forward neural network that is trained
    to minimize the quantile loss function

    .. math::
            \mathcal{L}_\tau(y_\tau, y_{true}) =
            \begin{cases} (1 - \tau)|y_\tau - y_{true}| & \text{ if } y_\tau < y_\text{true} \\
            \tau |y_\tau - y_\text{true}| & \text{ otherwise, }\end{cases}

    where :math:`x_\text{true}` is the expected value of the retrieval quantity
    and and :math:`x_\tau` is the predicted quantile. The neural network
    has one output neuron for each quantile to estimate.

    For the training, this implementation provides custom data generators that
    can be used to add Gaussian noise to the training data as well as adversarial
    training using the fast gradient sign method.

    This implementation also provides functionality to use an ensemble of networks
    instead of just a single network to predict the quantiles.

    .. note:: For the QRNN I am using :math:`x` to denote the input vector and
              :math:`y` to denote the output. While this is opposed to typical
              inverse problem notation, it is inline with machine learning
              notation and felt more natural for the implementation. If this
              annoys you, I am sorry. But the other way round it would have
              annoyed other people and in particular me.

    Attributes:

        input_dim (int):
            The input dimension of the neural network, i.e. the dimension of the
            measurement vector.

        quantiles (numpy.array):
            The 1D-array containing the quantiles :math:`\tau \in [0, 1]` that the
            network learns to predict.

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

    def __init__(self,
                 input_dim,
                 quantiles,
                 depth=3,
                 width=128,
                 activation="relu",
                 ensemble_size=1,
                 **kwargs):
        """
        Create a QRNN model.

        Arguments:

            input_dim(int): The dimension of the measurement space, i.e. the number
                            of elements in a single measurement vector y

            quantiles(np.array): 1D-array containing the quantiles  to estimate of
                                 the posterior distribution. Given as fractions
                                 within the range [0, 1].

            depth(int): The number of hidden layers  in the neural network to
                        use for the regression. Default is 3, i.e. three hidden
                        plus input and output layer.

            width(int): The number of neurons in each hidden layer.

            activation(str): The name of the activation functions to use. Default
                             is "relu", for rectified linear unit. See 
                             `this <https://keras.io/activations>`_ link for
                             available functions.

            **kwargs: Additional keyword arguments are passed to the constructor
                      call `keras.layers.Dense` of the hidden layers, which can
                      for example be used to add regularization. For more info consult
                      `Keras documentation. <https://keras.io/layers/core/#dense>`_
        """
        self.input_dim = input_dim
        self.quantiles = np.array(quantiles)
        self.depth = depth
        self.width = width
        self.activation = activation

        model = Sequential()
        if depth == 0:
            model.add(Dense(input_dim=input_dim,
                            units=len(quantiles),
                            activation=None))
        else:
            model.add(Dense(input_dim=input_dim,
                            units=width,
                            activation=activation))
            for i in range(depth - 2):
                model.add(Dense(units=width,
                                activation=activation,
                                **kwargs))
            model.add(Dense(units=len(quantiles), activation=None))
        self.models = [clone_model(model) for i in range(ensemble_size)]

    def __fit_params__(self, kwargs):
        at = kwargs.pop("adversarial_training", False)
        dat = kwargs.pop("delta_at", 0.01)
        batch_size = kwargs.pop("batch_size", 512)
        convergence_epochs = kwargs.pop("convergence_epochs", 10)
        initial_learning_rate = kwargs.pop('initial_learning_rate', 0.01)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 2.0)
        learning_rate_minimum = kwargs.pop('learning_rate_minimum', 1e-6)
        maximum_epochs = kwargs.pop("maximum_epochs", 200)
        training_split = kwargs.pop("training_split", 0.9)
        return at, dat, batch_size, convergence_epochs, initial_learning_rate, \
            learning_rate_decay, learning_rate_minimum, maximum_epochs, \
            training_split, kwargs

    def cross_validation(self,
                        x_train,
                        y_train,
                        sigma_noise = None,
                        n_folds=5,
                        s=None,
                        **kwargs):
        r"""
        Perform n-fold cross validation.

        This function trains the network n times on different subsets of the
        provided training data, always keeping a fraction of 1/n samples apart
        for testing. Performance for each of the networks is evaluated and mean
        and standard deviation for all folds are returned. This is to reduce
        the influence of random fluctuations of the network performance during
        hyperparameter tuning.

        Arguments:

            x_train(numpy.array): Array of shape :code:`(m, n)` containing the
                                  m n-dimensional training inputs.

            y_train(numpy.array): Array of shape :code:`(m, 1)` containing the
                                  m training outputs.

            sigma_noise(None, float, np.array): If not `None` this value is used
                                                to multiply the Gaussian noise
                                                that is added to each training
                                                batch. If None no noise is
                                                added.

            n_folds(int): Number of folds to perform for the cross correlation.

            s(callable, None): Performance metric for the fold. If not None,
                               this should be a function object taking as
                               arguments :code:`(y_test, y_pred)`, i.e. the
                               expected output for the given fold :code:`y_test`
                               and the predicted output :code:`y_pred`. The
                               returned value is taken as performance metric.

            **kwargs: Additional keyword arguments are passed on to the :code:`fit`
                      method that is called for each fold.
        """

        n = x_train.shape[0]
        n_test = n // n_folds
        inds = np.random.permutation(np.arange(0, n))

        results = []

        # Nomenclature is a bit difficult here:
        # Each cross validation fold has its own training,
        # vaildation and test set. The size of the test set
        # is number of provided training samples divided by the
        # number of fold. The rest is used a traning and internal
        # validation set according to the chose training_split
        # ratio.


        for i in range(n_folds):
            for m in self.models:
                m.reset_states()

            # Indices to use for training including training data and internal
            # validation set to monitor convergence.
            inds_train = np.append(np.arange(0, i * n_test),
                                       np.arange(min((i + 1) * n_test, n), n))
            inds_train = inds[inds_train]
            # Indices used to evaluate performance of the model.
            inds_test = np.arange(i * n_test, (i + 1) * n_test)
            inds_test = inds[inds_test]

            x_test_fold = x_train[inds_test, :]
            y_test_fold = y_train[inds_test]

            # Splitting training and validation set.
            x_train_fold = x_train[inds_train, :]
            y_train_fold = y_train[inds_train]

            self.fit(x_train_fold, y_train_fold,
                     sigma_noise, **kwargs)

            # Evaluation on this folds test set.
            if s:
                y_pred = self.predict(x_test_fold)
                results += [s(y_pred, y_test_fold)]
            else:
                loss = self.models[0].evaluate(
                    (x_test_fold - self.x_mean) / self.x_sigma,
                    y_test_fold)
                print(loss)
                results += [loss]
        print(results)
        results = np.array(results)
        print(results)
        return (np.mean(results, axis=0), np.std(results, axis=0))

    def fit(self,
            x_train,
            y_train,
            sigma_noise=None,
            x_val=None,
            y_val=None,
            **kwargs):
        r"""
        Train the QRNN on given training data.

        The training uses an internal validation set to monitor training
        progress. This can be either split at random from the training
        data (see `training_fraction` argument) or provided explicitly
        using the `x_val` and `y_val` parameters

        Training of the QRNN is performed using stochastic gradient descent
        (SGD). The learning rate is adaptively reduced during training when
        the loss on the internal validation set has not been reduced for a
        certain number of epochs.

        Two data augmentation techniques can be used to improve the
        calibration of the QRNNs predictions. The first one adds Gaussian
        noise to training batches before they are passed to the network.
        The noise statistics are defined using the `sigma_noise` argument.
        The second one is adversarial training. If adversarial training is
        used, half of each training batch consists of adversarial samples
        generated using the fast gradient sign method. The strength of the
        perturbation is controlled using the `delta_at` parameter.

        During one epoch, each training sample is passed once to the network.
        Their order is randomzied between epochs.

        Arguments:

            x_train(np.array): Array of shape `(n, m)` containing n training
                               samples of dimension m.

            y_train(np.array): Array of shape `(n, )` containing the training
                               output corresponding to the training data in
                               `x_train`.

            sigma_noise(None, float, np.array): If not `None` this value is used
                                                to multiply the Gaussian noise
                                                that is added to each training
                                                batch. If None no noise is
                                                added.
            x_val(np.array): Array of shape :code:`(n', m)` containing n' validation
                             inputs that will be used to monitor training loss. Must
                             be provided in unison with :code:`y_val` or otherwise
                             will be ignored.

            y_val(np.array): Array of shape :code:`(n')` containing n'  validation
                             outputs corresponding to the inputs in :code:`x_val`.
                             Must be provided in unison with :code:`x_val` or
                             otherwise will be ignored.

            adversarial_training(Bool): Whether or not to use adversarial training.
                                        `False` by default.

            delta_at(flaot): Perturbation factor for the fast gradient sign method
                             determining the strength of the adversarial training
                             perturbation. `0.01` by default.

            batch_size(float): The batch size to use during training. Defaults to `512`.

            convergence_epochs(int): The number of epochs without decrease in
                                     validation loss before the learning rate
                                     is reduced. Defaults to `10`.

            initial_learning_rate(float): The inital value for the learning
                                          rate.

            learning_rate_decay(float): The factor by which to reduce the
                                        learning rate after no improvement
                                        on the internal validation set was
                                        observed for `convergence_epochs`
                                        epochs. Defaults to `2.0`.

            learning_rate_minimum(float): The minimum learning rate at which
                                          the training is terminated. Defaults
                                          to `1e-6`.

            maximum_epochs(int): The maximum number of epochs to perform if
                                 training does not terminate before.

            training_split(float): The ratio `0 < ts < 1.0` of the samples in
                                   to be used as internal validation set. Defaults
                                   to 0.9.

        """
        if not (x_train.shape[1] == self.input_dim):
            raise Exception("Training input must have the same extent along"    /
                            "dimension 1 as input_dim (" + str(self.input_dim)  /
                            + ")")

        if not (y_train.shape[1] == 1):
            raise Exception("Currently only scalar retrieval targets are"       /
                            " supported.")

        x_mean = np.mean(x_train, axis=0, keepdims=True)
        x_sigma = np.std(x_train, axis=0, keepdims=True)
        self.x_mean = x_mean
        self.x_sigma = x_sigma

        # Handle parameters
        # at:  adversarial training
        # bs:  batch size
        # ce:  convergence epochs
        # ilr: initial learning rate
        # lrd: learning rate decay
        # lrm: learning rate minimum
        # me:  maximum number of epochs
        # ts:  split ratio of training set
        at, dat, bs, ce, ilr, lrd, lrm, me, ts, kwargs = self.__fit_params__(
            kwargs)

        # Split training and validation set if x_val or y_val
        # are not provided.
        n = x_train.shape[0]
        n_train = n
        if x_val is None and y_val is None:
            n_train = round(ts * n)
            n_val = n - n_train
            inds = np.random.permutation(n)
            x_val = x_train[inds[n_train:], :]
            y_val = y_train[inds[n_train:]]
            x_train = x_train[inds[:n_train], :]
            y_train = y_train[inds[:n_train]]
        loss = QuantileLoss(self.quantiles)

        self.custom_objects = {loss.__name__: loss}
        for model in self.models:
            optimizer = SGD(lr=ilr)
            model.compile(loss=loss, optimizer=optimizer)

            if at:
                inputs = [model.input, model.targets[0],
                          model.sample_weights[0]]
                input_gradients = K.function(
                    inputs, K.gradients(model.total_loss, model.input))
                training_generator = AdversarialTrainingGenerator(x_train,
                                                                  self.x_mean,
                                                                  self.x_sigma,
                                                                  y_train,
                                                                  sigma_noise,
                                                                  bs,
                                                                  input_gradients,
                                                                  dat)
            else:
                training_generator = TrainingGenerator(x_train, self.x_mean, self.x_sigma,
                                                       y_train, sigma_noise, bs)
            validation_generator = ValidationGenerator(x_val, self.x_mean, self.x_sigma,
                                                       y_val, sigma_noise)
            lr_callback = LRDecay(model, lrd, lrm, ce)
            model.fit_generator(training_generator, steps_per_epoch=n_train // bs,
                                epochs=me, validation_data=validation_generator,
                                validation_steps=1, callbacks=[lr_callback])

    def predict(self, x):
        r"""
        Predict quantiles of the conditional distribution P(y|x).

        Forward propagates the inputs in `x` through the network to
        obtain the predicted quantiles `y`.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` m-dimensional inputs
                         for which to predict the conditional quantiles.

        Returns:

             Array of shape `(n, k)` with the columns corresponding to the k
             quantiles of the network.

        """
        predictions = np.stack(
            [m.predict((x - self.x_mean) / self.x_sigma) for m in self.models])
        return np.mean(predictions, axis=0)

    def cdf(self, x):
        r"""
        Approximate the posterior CDF for given inputs `x`.

        Propagates the inputs in `x` forward through the network and
        approximates the posterior CDF by a piecewise linear function.

        The piecewise linear function is given by its values at
        approximate quantiles $x_\tau$ for
        :math: `\tau = \{0.0, \tau_1, \ldots, \tau_k, 1.0\}` where
        :math: `\tau_k` are the quantiles to be estimated by the network.
        The values for :math:`x_0.0` and :math:`x_1.0` are computed using

        .. math::
            x_0.0 = 2.0 x_{\tau_1} - x_{\tau_2}

            x_1.0 = 2.0 x_{\tau_k} - x_{\tau_{k-1}}

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.

        Returns:

            Tuple (xs, fs) containing the :math: `x`-values in `xs` and corresponding
            values of the posterior CDF :math: `F(x)` in `fs`.

        """
        y_pred = np.zeros(self.quantiles.size + 2)
        y_pred[1:-1] = self.predict(x)
        y_pred[0] = 2.0 * y_pred[1] - y_pred[2]
        y_pred[-1] = 2.0 * y_pred[-2] - y_pred[-3]

        qs = np.zeros(self.quantiles.size + 2)
        qs[1:-1] = self.quantiles
        qs[0] = 0.0
        qs[-1] = 1.0

        return y_pred, qs

    def sample_posterior(self, x, n=1):
        r"""
        Generates :code:`n` samples from the estimated posterior
        distribution for the input vector :code:`x`. The sampling
        is performed by the inverse CDF method using the estimated
        CDF obtained from the :code:`cdf` member function.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.

            n(int): The number of samples to generate.

        Returns:

            Tuple (xs, fs) containing the :math: `x`-values in `xs` and corresponding
            values of the posterior CDF :math: `F(x)` in `fs`.
        """
        y_pred, qs = self.cdf(x)
        p = np.random.rand(n)
        y = np.interp(p, qs, y_pred)
        return y

    @staticmethod
    def crps(y_pred, y_test, quantiles):
        r"""
        Compute the Continuous Ranked Probability Score (CRPS) for given quantile
        predictions.

        This function uses a piece-wise linear fit to the approximate posterior
        CDF obtained from the predicted quantiles in :code:`y_pred` to
        approximate the continuous ranked probability score (CRPS):

        .. math::
            CRPS(\mathbf{y}, x) = \int_{-\infty}^\infty (F_{x | \mathbf{y}}(x')
            - \mathrm{1}_{x < x'})^2 \: dx'

        Arguments:

            y_pred(numpy.array): Array of shape `(n, k)` containing the `k`
                                 estimated quantiles for each of the `n`
                                 predictions.

            y_test(numpy.array): Array containing the `n` true values, i.e.
                                 samples of the true conditional distribution
                                 estimated by the QRNN.

            quantiles: 1D array containing the `k` quantile fractions :math:`\tau`
                       that correspond to the columns in `y_pred`.

        Returns:

            `n`-element array containing the CRPS values for each of the
            predictions in `y_pred`.
        """
        y_cdf = np.zeros((y_pred.shape[0], quantiles.size + 2))
        y_cdf[:, 1:-1] = y_pred
        y_cdf[:, 0] = 2.0 * y_pred[:, 1] - y_pred[:, 2]
        y_cdf[:, -1] = 2.0 * y_pred[:, -2] - y_pred[:, -3]

        ind = np.zeros(y_cdf.shape)
        ind[y_cdf > y_test.reshape(-1, 1)] = 1.0

        qs = np.zeros((1, quantiles.size + 2))
        qs[0, 1:-1] = quantiles
        qs[0, 0] = 0.0
        qs[0, -1] = 1.0

        return np.trapz((qs - ind)**2.0, y_cdf)

    def evaluate_crps(self, x, y_test):
        r"""
        Predict quantiles and compute the Continuous Ranked Probability Score (CRPS).

        This function evaluates the networks prediction on the
        inputs in `x` and evaluates the CRPS of the predictions
        against the materializations in `y_test`.

        Arguments:

            x(numpy.array): Array of shape `(n, m)` containing the `n`
                            `m`-dimensional inputs for which to evaluate
                            the CRPS.

            y_test(numpy.array): Array containing the `n` materializations of
                                 the true conditional distribution.

        Returns:

            `n`-element array containing the CRPS values for each of the
            inputs in `x`.

        """
        return QRNN.crps(self.predict(x), y, self.quantiles)

    def save(self, path):
        r"""
        Store the QRNN model in a file.

        This stores the model to a file using pickle for all
        attributes that support pickling. The Keras model
        is handled separately, since it can not be pickled.

        .. note:: In addition to the model file with the given filename,
                  additional files suffixed with :code:`_model_i` will be
                  created for each neural network this model consists of.

        Arguments:

            path(str): The path including filename indicating where to
                       store the model.

        """

        f = open(path, "wb")
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        dirname = os.path.dirname(path)

        self.model_files = []
        for i, m in enumerate(self.models):
            self.model_files += [name + "_model_" + str(i)]
            m.save(os.path.join(dirname, self.model_files[i]))
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(path):
        r"""
        Load a model from a file.

        This loads a model that has been stored using the `save` method.

        Arguments:

            path(str): The path from which to read the model.

        Return:

            The loaded QRNN object.
        """
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        dirname = os.path.dirname(path)

        f = open(path, "rb")
        qrnn = pickle.load(f)
        qrnn.models = []
        for mf in qrnn.model_files:
            try:
                mp = os.path.join(dirname, mf)
                qrnn.models += [keras.models.load_model(mp, qrnn.custom_objects)]
            except(e):
                raise Exception("Error loading the neural network models. " \
                                "Please make sure all files created during the"\
                                " saving are in this folder.")
        f.close()
        return qrnn

    def __getstate__(self):
        dct = copy.copy(self.__dict__)
        dct.pop("models")
        return dct

    def __setstate__(self, state):
        self.__dict__ = state
        self.models = None
