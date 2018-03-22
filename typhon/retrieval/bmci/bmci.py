"""
Bayesian Montecarlo Integration.

Contains the `BMCI` class which implements the Bayesian Monte Carlo
Integration (BMCI) method as proposed by Evans et al. in [Evans]_.

.. [Evans] Evans, F. K. et al. Submillimeter-Wave Cloud Ice Radiometer: Simulations
   of retrieval algorithm performance. Journal of Geophysical Research 107, 2002
"""
import numpy as np

class BMCI:
    r"""
    Bayesian Monte Carlo Integration

    This class implements methods for solving Bayesian inverse problems using
    Monte Carlo integration. The method uses a data base of atmospheric states
    :math:`x_i` and corresponding  observations :math:`\mathbf{y}_i`, which are used
    to compute integrals of the posterior distribution by means of importance sampling:

    .. math::

        \int_a^b f(x') p(x' | \mathbf{y}) \: dx' \approx
        \sum_{a \leq x_i \leq b} \frac{w_i(\mathbf{y}) f(x)}{\sum_j w_j(\mathbf{y})}

    The measurements in the database are assumed to be given by a
    :math:`n \times m` matrix :math:`\mathbf{y}`, where n is the number of
    cases in the database. Currently only scalar retrievals are supported,
    which means that :math:`\mathbf{x}` is assumed to an be :math:`n`-element vector
    containing the retrieval quantities corresponding to the observations in
    :math:`\mathbf{y}`.

    The method assumes that the measurement uncertainty can be described by
    a zero-mean Gaussian distribution with covariance matrix :math:`\mathbf{S}_o`
    so that given an ideal forward model :math:`F: \mathrm{R}^n \rightarrow \mathrm{R}^m`
    the probability of a measurement :math:`\mathbf{y}` conditional on an
    atmospheric state :math:`\mathbf{x}` is proportional to

    .. math::
        P(\mathbf{y} | \mathbf{x}) \sim \exp
        \{ -\frac{1}{2} (\mathbf{y} - F(\mathbf{x}))^T
        \mathbf{S}_o^{-1} (\mathbf{y} - F(\mathbf{x})) \}

    Attributes

        x: numpy.array, shape = (n,)
           The retrieval quantity corresponding to the atmospheric states represented
           in the data base.

        y: numpy.array, shape = (n, m)
           The measured or simulated brightness temperatures corresponding to the
           atmospheric states represented in the data base.

        s_o: numpy.array, shape = (m, m)
           The covariance matrix describing the measurement uncertainty.

        n: int
           The number of entries in the retrieval database.

        m: int
           The number of channels in a single measurement :math:`\mathbf{y}_i`.

        pc1: numpy.array, shape = (m, )
            The eigenvector corresponding to the smallest eigenvalue of the
            observation uncertainty covariance matrix. Along this vector the
            entries in the database will be ordered, which can be used to
            accelerate the retrieval.

        pc1_e: float
            The smallest eigenvalue of the observation uncertainty covariance
            matrix.

        pc1_proj: numpy.array, shape (n, )
            The projections of the measurements in `y` onto `pc1` along which
            the database entries are ordered.

    """
    def __init__(self, y, x, s_o):
        """
        Create a QRNN instance from a given training data base
        `y, x` and measurement uncertainty given by the covariance
        matrix `s_o`.

        Args

            y(numpy.array): 2D array containing the measured or simulated
                            brightness temperatures corresponding to the
                            atmospheric states represented in the data base.
                            These are sorted along the eigenvector of the
                            observation uncertainty covariance matrix with
                            the smallest eigentvector.

            x: 1D array
            The retrieval quantity corresponding to the atmospheric states
            represented in the data base. These will be sorted along the
            eigenvector of the observation uncertainty covariance matrix
            with the smallest eigentvector.

            s_o: 2D array
            The covariance matrix describing the measurement uncertainty.
        """
        self.n = y.shape[0]
        self.m = y.shape[1]

        if not s_o.ndim == 2:
            raise Exception("Covariance matrix must be a 2D array.")
        if not self.m == s_o.shape[0] or not self.m == s_o.shape[1]:
            raise Exception("Covariance matrix must be square and agree "
                            + "with the dimensions of measurement vector.")

        # Covariance Matrix
        self.s_o = s_o
        self.s_o_inv = np.linalg.inv(self.s_o)


        # Eigenvalues of s
        self.y_mean  = np.mean(y, axis = 0)

        w, v = np.linalg.eig(self.s_o)

        inds = np.argsort(w)
        self.pc1_e = 1.0 / w[inds[0]]
        self.pc1 = v[:, inds[0]]
        self.pc1_proj = np.dot((y - self.y_mean), self.pc1)

        indices = np.argsort(self.pc1_proj)
        self.pc1_proj = self.pc1_proj[indices]
        self.x = x[indices]
        self.y = y[indices, :]

        self.x_sorted_inds = np.argsort(self.x)


    def __find_hits(self, y_obs, x2_max = 10.0):
        r"""
        Finds the range of indices in the database guaranteed to have a greater
        :math:`\chi^2` value than `x2_max`.

        Args:

            y_obs: 1D array
                   The measurement for which to compute the range of elements
                   to include in the weight calculation.

        Returns: Tuple of integers `(i_l, i_u, n_hits)`

            i_l: int
                 The lower index of the range of elements to include in the
                 weight calculation.

            i_u: int
                 The upper index of the range of elements to include in the
                 weight calculation.

            n_hits: int
                 The number of hits in the that are within the specified
                 :math:`\chi^2` limits.

        """
        y_proj = np.dot(self.pc1, (y_obs - self.y_mean).ravel())
        s_l = y_proj - np.sqrt(2.0 * x2_max / self.pc1_e)
        s_u = y_proj + np.sqrt(2.0 * x2_max / self.pc1_e)
        inds = np.searchsorted(self.pc1_proj, np.array([s_l, s_u]))

        return inds[0], inds[1], inds[1] - inds[0]

    def __gauss_prob(self, y_obs, y_database):

        dy = y_database - y_obs.reshape(1, -1)
        ws = dy * np.dot(dy, self.s_o_inv)
        ws = np.exp(-0.5 * ws.sum(axis=1, keepdims=True))
        return ws

    def weights(self, y_obs, x2_max = -1.0):
        r"""
        Compute the importance sampling weights for a given observation `y`.
        If `y_train` is given it will be used as the database observations
        for which to compute the weights. Thie can be used to reduce the lookup
        scope in order to improve computational performance.

        The weight :math:`w_i` for database entry :math:`i` and given
        observation :math:`y` is computed as:

        .. math::
            w_i = \exp \{ -\frac{1}{2} (\mathbf{y} - \mathbf{y}_i)^T
            \mathbf{S}_o^{-1}(\mathbf{y} - \mathbf{y}_i) \}

        Args:

            y(numpy.array): The observations for which to compute the weights.

            y_train(numpy.array): 2D array containing the observations from the
                                  database for which to compute the weights.
                                  Channels are assumed to be along axis 1.

        Returns:

            ws: 1D array
                Array containing the importance sampling weights.

        Return
        """
        if x2_max < 0.0:
            ws = self.__gauss_prob(y_obs, self.y )
            c  = ws.sum()
            return 0, self.n, ws
        else:
            i_l, i_u, n_hits = self.__find_hits(y_obs, x2_max)
            ws = self.__gauss_prob(y_obs, self.y[i_l : i_u, :])
            return i_l, i_u, ws

    def predict(self, y_obs, x2_max = -1.0):
        r"""
        This performs the BMCI integration to approximate the mean and variance
        of the posterior distribution:

        .. math::
            \int x p(x | \mathbf{y}) \: dx \approx
            \sum_i \frac{w_i(\mathbf{y}) x}{\sum_j w_j(\mathbf{y})}


            \int x^2 p(x | \mathbf{y}) \: dx \approx
            \sum_i \frac{w_i(\mathbf{y}) x^2}{\sum_j w_j(\mathbf{y})}

        If the keyword arg `x2_max` is provided, then the weights will be
        computed exluding database entries that are guaranteed to have a
        :math:`\chi^2` value higher than `x2_max`.

        Arguments:

            y_obs (numpy.ndarray): 2-D array containing the observations for
                                   which to perform the retrieval.

        Returns:

            A tuple `(xs, sigmas)` containing the retrieved means (`xs`) and
            the corresponding standard deviations (`sigmas`).

        """
        xs = np.zeros(y_obs.shape[0])
        sigmas = np.zeros(y_obs.shape[0])

        for i in range(y_obs.shape[0]):
            i_l, i_u, ws = self.weights(y_obs[i, :], x2_max)
            c = ws.sum()
            if c > 0.0:
                xs[i] = np.sum(self.x[i_l:i_u].ravel() * ws.ravel() / c)
                sigmas[i] = np.sqrt(np.sum(
                    (self.x[i_l:i_u].ravel() - xs[i]) ** 2.0 * ws.ravel() / c))
            else:
                xs[i] = np.float("nan")
                sigmas[i] = np.float("nan")
        return xs, sigmas

    def crps(self, y_obs, x_true, x2_max = -1.0):
        r"""

        This function approximates the cumulative probability density of the
        posterior distribution for the given observations in `y_obs` and
        computes the continuously ranked probability score:

        .. math::
            CRPS(\mathbf{y}, x) = \int_{-\infty}^\infty (F_{x | \mathbf{y}}(x')
            - \mathrm{1}_{x < x'})^2 \: dx'

        Arguments:

            y_obs(numpy.ndarray): 2-D array with shape `(n, m)` containing the
                                   n observations for which to evaluate the CRPS.
            x_true(numpy.ndarray): 1-D array containing the `n` x values to test
                                   the predictions against.

        """
        n = y_obs.shape[0]
        scores = np.zeros(n)

        for i in range(n):
            i_l, i_u, ws = self.weights(y_obs[i, :], x2_max)

            inds = np.where((i_l <= self.x_sorted_inds) * (self.x_sorted_inds < i_u))
            inds = self.x_sorted_inds[inds] - i_l

            ws = ws[inds]
            xs = self.x[i_l:i_u][inds]

            indicator = np.zeros(i_u - i_l)
            indicator[xs > x_true[i]] = 1.0

            ws_cum = ws.cumsum()

            if ws_cum[-1] > 0.0:
                ws_cum /= ws_cum[-1]
                scores[i] = np.trapz((ws_cum - indicator) ** 2.0,
                                     xs)
            else:
                scores[i] = np.float("nan")

        return scores

    def cdf(self, y_obs, x2_max = -1):
        r"""
        Cumulative posterior density function.

        This function approximates the cumulative posterior distribution
        :math:`F(x | \mathbf{y})` for the given observation `y_obs` using

        .. math::

            F(x | \mathbf{y}) = \int_{-\infty}^{x} p(x' | \mathbf{y}) \: dx'
            \approx \sum_{x_i < x} \frac{w_i(\mathbf{y})}{\sum_j w_j(\mathbf{y})}

        Args:

            y_obs(numpy.array): `m`-element array containing the observation
                                 for which to compute the posterior CDF.

            x2_max(float): The :math:`\chi^2` cutoff to apply to elements in the
                           database. Ignored if less than zero.

        Returns:

            A tuple `(xs, ys)` containing the estimated values of the posterior
            CDF :math:`F(x | \mathbf{y})` evaluated at the $x$ values
            corresponding to the hits in the database.

        Raises:

            ValueError
                If the number of channels in the observations is different from
                the database.

        """
        try:
            y_obs = y_obs.reshape(1, self.m)
        except:
            raise ValueError("The observation vector is inconsistent"
                             "with the database.")

        i_l, i_u, ws = self.weights(y_obs, x2_max)

        inds = np.where((i_l <= self.x_sorted_inds) * (self.x_sorted_inds < i_u))
        inds = self.x_sorted_inds[inds] - i_l

        ws = ws[inds]
        xs = self.x[i_l:i_u][inds]

        ws_cum = ws.cumsum()
        if ws_cum[-1] > 0.0:
            ws_cum /= ws_cum[-1]
        else:
            ws_cum = np.float("nan")
        return xs, ws_cum

    def predict_quantiles(self, y_obs, taus, x2_max = -1):
        r"""
        This estimates the quantiles given in `taus` by approximating
        the CDF of the posterior as

        .. math::

            F(x | \mathbf{y}) = \int_{-\infty}^{x} p(x' | \mathbf{y}) \: dx'
            \approx \sum_{x_i < x} \frac{w_i(\mathbf{y})}{\sum_j w_j(\mathbf{y})}

        and then interpolating :math:`F^{-1}` to obtain the desired quantiles.

        Args:

            y_obs(numpy.array): `n`-times-`m` matrix containing the `n`
                                 observations with `m` channels for which to
                                 compute the percentiles.

            taus(numpy.array): 1D array containing the `k` quantiles
                               :math:`\tau \in [0,1]` to compute.

            x2_max(float): The :math:`\chi^2` cutoff to apply to elements in the
                           database. Ignored if less than zero.

        Returns:

            A 2D numpy.array with shape `(n, k)` array containing the estimated
            quantiles or `NAN` if no database entries were found in the
            :math:`\chi^2` search  region.

        Raises:

            ValueError
                If the number of channels in the observations is different from
                the database.

            ValueError
                If any of the percentiles lies outside the interval [0, 1].

        """
        taus = np.asarray(taus).reshape((-1, ))

        m = y_obs.shape[1]
        n = y_obs.shape[0]
        k = taus.size

        if not m == self.m:
            raise ValueError("Number of channels is inconsistent with database.")

        if np.any((taus < 0.0) + (taus > 1.0)):
            raise ValueError("Percentiles must be in [0.0, 1.0]")

        qs = np.zeros((n, k))
        for i in range(y_obs.shape[0]):

            i_l, i_u, ws = self.weights(y_obs[i, :], x2_max)

            inds = np.where((i_l <= self.x_sorted_inds) * (self.x_sorted_inds < i_u))
            inds = self.x_sorted_inds[inds] - i_l

            ws = ws[inds]
            xs = self.x[i_l:i_u][inds]

            ws_cum = ws.cumsum()

            if ws_cum[-1] > 0.0:
                ws_cum /= ws_cum[-1]
                qs[i, :] = np.interp(taus, ws_cum, xs)
            else:
                qs[i, :] = np.float("nan")
        return qs
