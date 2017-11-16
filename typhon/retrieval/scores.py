r"""
This module contains functions for the computations of scores for assessment
of retrieval methods.
"""
import numpy as np


def mape(y_pred, y_test):
    r"""
    The Mean Absolute Percentage Error (MAPE)

    The MAPE is computed as the mean of the absolute value of the relative
    error in percent, i.e.:

    .. math::

        \text{MAPE}(\mathbf{y}, \mathbf{y}_{true}) =
        \frac{100\%}{n}\sum_{i = 0}^n \frac{|y_{\text{pred},i} - y_{\text{true},i}|}
             {|y_{\text{true},i}|}

    Arguments:

        y_pred(numpy.array): The predicted scalar values.

        y_test(numpy.array): The true values.

    Returns:

        The MAPE for the given predictions.

    """
    return np.nanmean(100.0 * np.abs(y_test - y_pred.ravel()) / np.abs(y_test).ravel())


def bias(y_pred, y_test):
    r"""
    The mean bias in percent.

    .. math::

        \text{BIAS}(\mathbf{y}, \mathbf{y}_{true}) =
        \frac{100\%}{n}\sum_{i = 0}^n \frac{y_{\text{pred},i} - y_{\text{true},i}}
             {y_{\text{true},i}}

    Arguments:

        y_pred(numpy.array): The predicted scalar values.

        y_test(numpy.array): The true values.

    Returns:

        The mean bias in percent.

    """
    return np.mean(100.0 * y_test - y_pred / y_test)


def quantile_score(y_tau, y_test, taus):
    r"""
    The quantile loss function.

    Let $y_\tau$ be an estimate of the $\tau\text{th}$ quantile of a
    random distribution $Y$. Then the quantile loss is a proper scoring
    rule that measures the quality of the estimate $y_\tau$ given a
    sample $y$ from $Y$:

    .. math::
            \mathcal{L}_\tau(y_\tau, y_{true}) =
            \begin{cases} (1 - \tau)|y_\tau - y_{true}| & \text{ if } y_\tau < y_\text{true} \\
            \tau |y_\tau - y_\text{true}| & \text{ otherwise. }\end{cases}

    Arguments:

        y_tau(numpy.array): Numpy array with shape (n, k) containing one row of
                            k estimated quantiles for each of the n test cases.

        y_test(numpy.array): Numpy array with the n observed test values
                             of the conditional distributions whose quantiles are
                             estimated by the elements in `y_tau`

        taus(numpy.array): Numpy array containing the k quantile fractions
                           :math:`\tau` that are estimated by the columns in
                           `y_tau`.

    Returns:

        Array of shape (n, k) containing the quantile scores for each quantile
        estimate.

    Raises:

        ValueError
            If the shapes of `y_tau`, `y_test` and `taus` are inconsistent.
    """
    taus = np.asarray(taus)
    m = taus.size

    y_tau = y_tau.reshape(-1, m)
    n = y_tau.shape[0]

    try:
        y_test = y_test.reshape(n, 1)
    except:
        raise ValueError(
            "Shape of y_test is incompatible with y_tau and taus.")

    abs_1 = taus * np.abs(y_tau - y_test)
    abs_2 = (1.0 - taus) * np.abs(y_tau - y_test)

    return np.where(y_tau < y_test, abs_1, abs_2)


def mean_quantile_score(y_tau, y_test, taus):
    r"""
    Wrapper around the `quantile_score` function, which computes the mean
    along the first dimension.
    """
    return np.nanmean(quantile_score(y_tau, y_test, taus), axis=0)
