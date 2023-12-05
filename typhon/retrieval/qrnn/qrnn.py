"""
typhon.retrieval.qrnn.qrnn
==========================

This module provides the QRNN class, which implements the high-level
functionality of quantile regression neural networks, while the neural
network implementation is left to the model backends implemented in the
``typhon.retrieval.qrnn.models`` submodule.
"""
import copy
import logging
import os
import pickle
import importlib
import scipy
from scipy.stats import norm

import numpy as np
from scipy.interpolate import CubicSpline

################################################################################
# Set the backend
################################################################################

try:
    import typhon.retrieval.qrnn.models.keras as keras
    backend = keras
except Exception as e:
    try:
        import typhon.retrieval.qrnn.models.pytorch as pytorch
        backend = pytorch
    except:
        raise Exception("Couldn't import neither Keras nor Pytorch "
                        "one of them must be available to use the QRNN"
                        " module.")

def set_backend(name):
    """
    Set the neural network package to use as backend.

    The currently available backend are "keras" and "pytorch".

    Args:
        name(str): The name of the backend.
    """
    global backend
    if name == "keras":
        try:
            import typhon.retrieval.qrnn.models.keras as keras
            backend = keras
        except Exception as e:
            raise Exception("The following error occurred while trying "
                            " to import keras: ", e)
    elif name == "pytorch":
        try:
            import typhon.retrieval.qrnn.models.pytorch as pytorch
            backend = pytorch
        except Exception as e:
            raise Exception("The following error occurred while trying "
                            " to import pytorch: ", e)
    else:
        raise Exception("\"{}\" is not a supported backend.".format(name))

def get_backend(name):
    """
    Get module object corresponding to the short backend name.

    The currently available backend are "keras" and "pytorch".

    Args:
        name(str): The name of the backend.
    """
    if name == "keras":
        try:
            import typhon.retrieval.qrnn.models.keras as keras
            backend = keras
        except Exception as e:
            raise Exception("The following error occurred while trying "
                            " to import keras: ", e)
    elif name == "pytorch":
        try:
            import typhon.retrieval.qrnn.models.pytorch as pytorch
            backend = pytorch
        except Exception as e:
            raise Exception("The following error occurred while trying "
                            " to import pytorch: ", e)
    else:
        raise Exception("\"{}\" is not a supported backend.".format(name))
    return backend

def fit_gaussian_to_quantiles(y_pred, taus):
    """
    Fits Gaussian distributions to predicted quantiles.

    Fits mean and standard deviation values to quantiles by minimizing
    the mean squared distance of the predicted quantiles and those of
    the corresponding Gaussian distribution.

    Args:
        y_pred (``np.array``): Array of shape `(n, m)` containing the `m`
            predicted quantiles for n different inputs.
        taus(``np.array``): Array of shape `(m,)` containing the quantile
            fractions corresponding to the predictions in ``y_pred``.

    Returns:
        Tuple ``(mu, sigma)`` of vectors of size `n` containing the mean and
        standard deviations of the Gaussian distributions corresponding to
        the predictions in ``y_pred``.
    """
    x = norm.ppf(taus)

    d2e_00 = x.size
    d2e_01 = x.sum(axis=-1)
    d2e_10 = x.sum(axis=-1)
    d2e_11 = np.sum(x ** 2, axis=-1)

    d2e_det_inv = 1.0 / (d2e_00 * d2e_11 - d2e_01 * d2e_11)
    d2e_inv_00 = d2e_det_inv * d2e_11
    d2e_inv_01 = -d2e_det_inv * d2e_01
    d2e_inv_10 = -d2e_det_inv * d2e_10
    d2e_inv_11 = d2e_det_inv * d2e_00

    de_0 = -np.sum(y_pred - x, axis=-1)
    de_1 = -np.sum(x * (y_pred - x), axis=-1)

    mu = -(d2e_inv_00 * de_0 + d2e_inv_01 * de_1)
    sigma = 1.0 - (d2e_inv_10 * de_0 + d2e_inv_11 * de_1)

    return mu, sigma

def create_model(input_dim,
                 output_dim,
                 arch):
    """
    Creates a fully-connected neural network from a tuple
    describing its architecture.

    Args:
        input_dim(int): Number of input features.
        output_dim(int): Number of output features.
        arch: Tuple (d, w, a) containing the depth, i.e. number of
            hidden layers width of the hidden layers, i. e.
            the number of neurons in them, and the name of the
            activation function as string.
    Return:
        Depending on the available backends, a fully-connected
        keras or pytorch model, with the requested number of hidden
        layers and neurons in them.
    """
    return backend.FullyConnected(input_dim, output_dim, arch)


################################################################################
# QRNN class
################################################################################

class QRNN:
    r"""
    Quantile Regression Neural Network (QRNN)

    This class provides a high-level implementation of  quantile regression
    neural networks. It can be used to estimate quantiles of the posterior
    distribution of remote sensing retrievals.

    The :class:`QRNN`` class uses an arbitrary neural network model, that is
    trained to minimize the quantile loss function

    .. math::
            \mathcal{L}_\tau(y_\tau, y_{true}) =
            \begin{cases} (1 - \tau)|y_\tau - y_{true}| & \text{ if } y_\tau < y_\text{true} \\
            \tau |y_\tau - y_\text{true}| & \text{ otherwise, }\end{cases}

    where :math:`x_\text{true}` is the true value of the retrieval quantity
    and and :math:`x_\tau` is the predicted quantile. The neural network
    has one output neuron for each quantile to estimate.

    The QRNN class provides a generic QRNN implementation in the sense that it
    does not assume a fixed neural network architecture or implementation.
    Instead, this functionality is off-loaded to a model object, which can be
    an arbitrary regression network such as a fully-connected or a
    convolutional network. A range of different models are provided in the
    typhon.retrieval.qrnn.models module. The :class:`QRNN`` class just
    implements high-level operation on the QRNN output while training and
    prediction are delegated to the model object. For details on the respective
    implementation refer to the documentation of the corresponding model class.

    .. note::

      For the QRNN implementation :math:`x` is used to denote the input
      vector and :math:`y` to denote the output. While this is opposed
      to inverse problem notation typically used for retrievals, it is
      in line with machine learning notation and felt more natural for
      the implementation. If this annoys you, I am sorry.

    Attributes:
        backend(``str``):
            The name of the backend used for the neural network model.
        quantiles (numpy.array):
            The 1D-array containing the quantiles :math:`\tau \in [0, 1]`
            that the network learns to predict.
        model:
            The neural network regression model used to predict the quantiles.
    """
    def __init__(self,
                 input_dimensions,
                 quantiles=None,
                 model=(3, 128, "relu"),
                 ensemble_size=1,
                 **kwargs):
        """
        Create a QRNN model.

        Arguments:
            input_dimensions(int):
                The dimension of the measurement space, i.e. the
                number of elements in a single measurement vector y
            quantiles(np.array):
                1D-array containing the quantiles  to estimate of
                the posterior distribution. Given as fractions within the range
                [0, 1].
            model:
                A (possibly trained) model instance or a tuple
                ``(d, w, act)`` describing the architecture of a fully-connected
                neural network with :code:`d` hidden layers with :code:`w` neurons
                and :code:`act` activation functions.
        """
        self.input_dimensions = input_dimensions
        self.quantiles = np.array(quantiles)
        self.backend = backend.__name__

        if type(model) == tuple:
            self.model = backend.FullyConnected(self.input_dimensions,
                                                self.quantiles,
                                                model)
            if quantiles is None:
                raise ValueError("If model is given as architecture tuple, the"
                                  " 'quantiles' kwarg must be provided.")
        else:
            if not quantiles is None:
                if not quantiles == model.quantiles:
                    raise ValueError("Provided quantiles do not match those of "
                                     "the provided model.")

            self.model = model
            self.quantiles = model.quantiles
            self.backend = model.backend

    def train(self,
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
              gpu = False):
        """
        Train model on given training data.

        The training is performed on the provided training data and an
        optionally-provided validation set. Training can use the following
        augmentation methods:

        - Gaussian noise added to input
        - Adversarial training

        The learning rate is decreased gradually when the validation or training
        loss did not decrease for a given number of epochs.

        Args:
            training_data: Tuple of numpy arrays of a dataset object to use to
                train the model.
            validation_data: Optional validation data in the same format as the
                training data.
            batch_size: If training data is provided as arrays, this batch size
                will be used to for the training.
            sigma_noise: If training data is provided as arrays, training data
                will be augmented by adding noise with the given standard
                deviations to each input vector before it is presented to the
                model.
            adversarial_training(``bool``): Whether or not to perform
                adversarial training using the fast gradient sign method.
            delta_at: The scaling factor to apply for adversarial training.
            initial_learning_rate(``float``): The learning rate with which the
                 training is started.
            momentum(``float``): The momentum to use for training.
            convergence_epochs(``int``): The number of epochs with
                 non-decreasing loss before the learning rate is decreased
            learning_rate_decay(``float``): The factor by which the learning rate
                 is decreased.
            learning_rate_minimum(``float``): The learning rate at which the
                 training is aborted.
            maximum_epochs(``int``): For how many epochs to keep training.
            training_split(``float``): If no validation data is provided, this
                 is the fraction of training data that is used for validation.
            gpu(``bool``): Whether or not to try to run the training on the GPU.
        """
        return self.model.train(training_data,
                                validation_data,
                                batch_size,
                                sigma_noise,
                                adversarial_training,
                                delta_at,
                                initial_learning_rate,
                                momentum,
                                convergence_epochs,
                                learning_rate_decay,
                                learning_rate_minimum,
                                maximum_epochs,
                                training_split,
                                gpu)

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
        return self.model.predict(x)

    def cdf(self, x):
        r"""
        Approximate the posterior CDF for given inputs `x`.

        Propagates the inputs in `x` forward through the network and
        approximates the posterior CDF by a piecewise linear function.

        The piecewise linear function is given by its values at approximate
        quantiles :math:`x_\tau`` for :math:`\tau = \{0.0, \tau_1, \ldots,
        \tau_k, 1.0\}` where :math:`\tau_k` are the quantiles to be estimated
        by the network. The values for :math:`x_{0.0}` and :math:`x_{1.0}` are
        computed using

        .. math::

            x_{0.0} = 2.0 x_{\tau_1} - x_{\tau_2}

            x_{1.0} = 2.0 x_{\tau_k} - x_{\tau_{k-1}}

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.

        Returns:

            Tuple (xs, fs) containing the :math:`x`-values in `xs` and corresponding
            values of the posterior CDF :math:`F(x)` in `fs`.

        """
        if len(x.shape) > 1:
            s = x.shape[:-1] + (self.quantiles.size + 2,)
        else:
            s = (1, self.quantiles.size + 2)

        y_pred = np.zeros(s)
        y_pred[:, 1:-1] = self.predict(x)
        y_pred[:, 0] = 2.0 * y_pred[:, 1] - y_pred[:, 2]
        y_pred[:, -1] = 2.0 * y_pred[:, -2] - y_pred[:, -3]

        qs = np.zeros(self.quantiles.size + 2)
        qs[1:-1] = self.quantiles
        qs[0] = 0.0
        qs[-1] = 1.0

        return y_pred, qs

    def calibration(self, *args, **kwargs):
        """
        Compute calibration curve for the given dataset.
        """
        return self.model.calibration(*args, *kwargs)

    def pdf(self, x):
        r"""
        Approximate the posterior probability density function (PDF) for given
        inputs ``x``.

        The PDF is approximated by computing the derivative of the piece-wise
        linear approximation of the CDF as computed by the
        :py:meth:`typhon.retrieval.qrnn.QRNN.cdf` function.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
               to predict PDFs.

        Returns:

            Tuple (x_pdf, y_pdf) containing the array with shape `(n, k)`  containing
            the x and y coordinates describing the PDF for the inputs in ``x``.

        """
        x_cdf, y_cdf = self.cdf(x)
        n, m = x_cdf.shape
        x_pdf = np.zeros((n, m + 1))

        x_pdf[:, 0] = x_cdf[:, 0]
        x_pdf[:, -1] = x_cdf[:, -1]
        x_pdf[:, 1:-1] = 0.5 * (x_cdf[:, 1:] + x_cdf[:, :-1])

        y_pdf = np.zeros((n, m + 1))
        y_pdf[:, 1:-1] = np.diff(y_cdf) / np.diff(x_cdf, axis=-1)
        return x_pdf, y_pdf

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

            Tuple (xs, fs) containing the :math:`x`-values in `xs` and corresponding
            values of the posterior CDF :math: `F(x)` in `fs`.
        """
        result = np.zeros((x.shape[0], n))
        x_cdf, y_cdf = self.cdf(x)
        for i in range(x_cdf.shape[0]):
            p = np.random.rand(n)
            y = np.interp(p, y_cdf, x_cdf[i, :])
            result[i, :] = y
        return result

    def sample_posterior_gaussian_fit(self, x, n=1):
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

            Tuple (xs, fs) containing the :math:`x`-values in `xs` and corresponding
            values of the posterior CDF :math: `F(x)` in `fs`.
        """
        result = np.zeros((x.shape[0], n))
        y_pred = self.predict(x)
        mu, sigma = fit_gaussian_to_quantiles(y_pred, self.quantiles)
        x = np.random.normal(size=(y_pred.shape[0], n))
        return mu.reshape(-1, 1) + sigma.reshape(-1, 1) * x

    def posterior_mean(self, x):
        r"""
        Computes the posterior mean by computing the first moment of the
        estimated posterior CDF.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the posterior mean.
        Returns:

            Array containing the posterior means for the provided inputs.
        """
        y_pred, qs = self.cdf(x)
        mus = y_pred[:, -1] - np.trapz(qs, x=y_pred)
        return mus

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

    def evaluate_crps(self, x_test, y_test):
        r"""
        Predict quantiles and compute the Continuous Ranked Probability Score (CRPS).

        This function evaluates the network predictions on the test data
        ``x_test`` and ``y_test`` and and evaluates the CRPS.

        Arguments:

            x_test(numpy.array): Array of shape `(n, m)` input test data.

            y_test(numpy.array): Array of length n containing the output test
                 data.

        Returns:

            `n`-element array containing the CRPS values for each of the
            inputs in `x`.

        """
        return QRNN.crps(self.predict(x_test), y_test, self.quantiles)

    def classify(self, x, threshold):
        """
        Classify output based on posterior PDF and given numeric threshold.

        Args:
            x: The input data as :code:`np.ndarray` or backend-specific
               dataset object.
            threshold: The numeric threshold to apply for classification.
        """
        y = self.predict(x)
        out_shape = y.shape[:1] + (1,) + y.shape[2:]
        c = self.quantiles[0] * np.ones(out_shape)

        for i in range(self.quantiles.size - 1):
            q_l = y[:, [i]]
            q_r = y[:, [i+1]]
            inds = np.logical_and(q_l < threshold,
                                  q_r >= threshold)
            c[inds] = self.quantiles[i] * (threshold - q_l[inds])
            c[inds] += self.quantiles[i + 1] * (q_r[inds] - threshold)
            c[inds] /= (q_r[inds] - q_l[inds])

        c[threshold > q_r] = self.quantiles[-1]
        return 1.0 - c

    @staticmethod
    def load(path):
        r"""
        Load a model from a file.

        This loads a model that has been stored using the
        :py:meth:`typhon.retrieval.qrnn.QRNN.save`  method.

        Arguments:

            path(str): The path from which to read the model.

        Return:

            The loaded QRNN object.
        """
        with open(path, 'rb') as f:
            qrnn = pickle.load(f)
            backend = importlib.import_module(qrnn.backend)
            model = backend.load_model(f, qrnn.quantiles)
            qrnn.model = model
        return qrnn

    def save(self, path):
        r"""
        Store the QRNN model in a file.

        This stores the model to a file using pickle for all attributes that
        support pickling. The Keras model is handled separately, since it can
        not be pickled.

        Arguments:

            path(str): The path including filename indicating where to
                       store the model.

        """
        f = open(path, "wb")
        pickle.dump(self, f)
        backend = importlib.import_module(self.backend)
        backend.save_model(f, self.model)
        f.close()


    def __getstate__(self):
        dct = copy.copy(self.__dict__)
        dct.pop("model")
        return dct

    def __setstate__(self, state):
        self.__dict__ = state
        self.models = None
