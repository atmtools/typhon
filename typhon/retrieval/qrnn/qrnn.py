import copy
import logging
import os
import pickle

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
    does not assume a fixed neural network infrastructure. Instead, this
    functionality is off-loaded to a model object, which can be an arbitrary
    regression network such as a fully-connected or a convolutional network. A
    range of different models are provided in the typhon.retrieval.qrnn.models
    module. The :class:`QRNN`` class just implements high-level operation on
    the QRNN output while training and prediction are delegated to the model
    object. For details on the respective implementation refer to the
    documentation of the corresponding model class.

    .. note:: For the QRNN implementation :math:`x` is used to denote the input
              vector and :math:`y` to denote the output. While this is opposed
              to inverse problem notation typically used for retrievals, it is
              in line with machine learning notation and felt more natural for
              the implementation. If this annoys you, I am sorry.

    Attributes:
        quantiles (numpy.array): The 1D-array containing the quantiles
            :math:`\tau \in [0, 1]` that the network learns to predict.
        model: The model object that implements the actual neural network
            regressor.
    """
    def __init__(self,
                 input_dimension,
                 quantiles,
                 model=(3, 128, "relu"),
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
            model: A (possible trained) model instance or a tuple
                :code:`(d, w, act)` describing the architecture of a
                fully-connected neural network with :code:`d` hidden layers
                with :code:`w` neurons and :code:`act` activation functions.
            ensemble_size: The size of the ensemble if an ensemble model
                should be used.
        """
        self.input_dimension = input_dimension
        self.quantiles = np.array(quantiles)
        self.backend = backend.__name__

        if type(model) == tuple:
            self.model = backend.FullyConnected(self.input_dimension,
                                                self.quantiles,
                                                model)

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
                logger.info(loss)
                results += [loss]
        logger.info(results)
        results = np.array(results)
        logger.info(results)
        return (np.mean(results, axis=0), np.std(results, axis=0))

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

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

    def calibration(self, *args, **kwargs):
        """
        Compute calibration curve for the given dataset.
        """
        return self.model.calibration(*args, *kwargs)

    def pdf(self, x, use_splines = False):
        r"""
        Approximate the posterior probability density function (PDF) for given
        inputs `x`.

        By default, the PDF is approximated by computing the derivative of the
        piece-wise linear approximation of the CDF as computed by the :code:`cdf`
        function.

        If :code:`use_splines` is set to :code:`True`, the PDF is computed from
        a spline fit to the approximate CDF.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.

            use_splines(bool): Whether or not to use a spline fit to the CDF to
            approximate the PDF.

        Returns:

            Tuple (xs, fs) containing the :math: `x`-values in `xs` and corresponding
            values of the approximate posterior PDF :math: `F(x)` in `fs`.

        """

        y_pred = np.zeros(self.quantiles.size)
        y_pred = self.predict(x).ravel()

        y = np.zeros(y_pred.size + 1)
        y[1:-1] = 0.5 * (y_pred[1:] + y_pred[:-1])
        y[0] = 2 * y_pred[0] - y_pred[1]
        y[-1] = 2 * y_pred[-1] - y_pred[-2]

        if not use_splines:

            p = np.zeros(y.size)
            p[1:-1] = np.diff(self.quantiles) / np.diff(y_pred)
        else:

            y = np.zeros(y_pred.size + 2)
            y[1:-1] = y_pred
            y[0] = 3 * y_pred[0] - 2 * y_pred[1]
            y[-1] = 3 * y_pred[-1] - 2 * y_pred[-2]
            q = np.zeros(self.quantiles.size + 2)
            q[1:-1] = np.array(self.quantiles)
            q[0] = 0.0
            q[-1] = 1.0

            sr = CubicSpline(y, q, bc_type = "clamped")
            y = np.linspace(y[0], y[-1], 101)
            p = sr(y, nu = 1)

        return y, p


        y_pred = np.zeros(self.quantiles.size + 2)
        y_pred[1:-1] = self.predict(x)
        y_pred[0] = 2.0 * y_pred[1] - y_pred[2]
        y_pred[-1] = 2.0 * y_pred[-2] - y_pred[-3]

        if use_splines:
            x_t = np.zeros(x.size + 2)
            x_t[1:-1] = x
            x_t[0] = 2 * x[0] - x[1]
            x_t[-1] = 2 * x[-1] - x[-2]
            y_t = np.zeros(y.size + 2)
            y_t[1:-1] = y
            y_t[-1] = 1.0

        else:
            logger.info(y)
            x_new = np.zeros(x.size - 1)
            x_new[2:-2] = 0.5 * (x[2:-3] + x[3:-2])
            x_new[0:2] = x[0:2]
            x_new[-2:] = x[-2:]
            y_new = np.zeros(y.size - 1)
            y_new[1:-1] = np.diff(y[1:-1]) / np.diff(x[1:-1])
        return x_new, y_new

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
        mus = y_pred[-1] - np.trapz(qs, x=y_pred)
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
        return QRNN.crps(self.predict(x), y_test, self.quantiles)

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
        pickle.dump(self, f)
        self.model.save(f)
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
        import importlib
        with open(path, 'rb') as f:
            qrnn = pickle.load(f)
            backend = importlib.import_module(qrnn.backend)
            model = backend.load(f)
            qrnn.model = model
        return qrnn

        #try:
        #    from typhon.retrieval.qrnn.backends.keras import KerasQRNN, QuantileLoss
        #    globals()["QuantileLoss"] = QuantileLoss
        #    model = KerasQRNN.load(path)
        #    print(type(model))
        #    qrnn = QRNN(model.input_dim, model.quantiles, model.models)
        #    qrnn.model = model
        #    globals().pop("QuantileLoss")
        #    return qrnn

        #except Exception as e:
        #    raise e

    def __getstate__(self):
        dct = copy.copy(self.__dict__)
        dct.pop("model")
        return dct

    def __setstate__(self, state):
        self.__dict__ = state
        self.models = None
