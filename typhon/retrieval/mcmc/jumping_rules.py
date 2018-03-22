"""
Jumping rules for MCMC
======================

This module contains classes that implement jumping rules for MCMC.

"""
import numpy as np

class RandomWalk:
    """
    Random walk jumping function for MCMC.
    """
    def __init__(self, covmat):
        """
        Create a random walker with given covariance.

        Args:
            covmat: A square, 2D numpy.array that should be used as covariance
            matrix for the random walk.
        """
        try:
            self.covmat = np.asarray(covmat)

            # covmat must be a matrix.
            if not self.covmat.ndim == 2:
                raise Exception("Covariance matrix must be 2 dimensional.")

            # covmat must be square.
            self.n = covmat.shape[0]
            if not self.n == covmat.shape[1]:
                raise Exception("Covariance matrix must be square.")

        except:
            raise Exception("covmat argument must be convertible to a numpy array.")

    def step(self, x):
        """
        Generates a new step x_new from the current position x.

        Args:
            x: The current position of the random walker.
        """
        return np.random.multivariate_normal(x.ravel(), self.covmat).reshape(x.shape)

    def update(self, hist):
        """
        Update covariance from a sequence of samples.

        This computes the covariance matrix of a given sequence of samples,
        scales it by  2.4/sqrt(n))^2 and sets it to the covariance matrix
        to be used for the random walk.

        Args:
            hist: 2D numpy array with shape (m,n) where m is number of steps in
            the sequence and n is the number of dimensions of the parameter
            space.
        """
        if not hist.shape[1] == self.n:
            raise Exception("Provided array does not have the expected dimensions.")
        mean = np.mean(hist, axis=0, keepdims=True)
        d = hist - mean
        s = np.dot(np.transpose(d), d) / hist.shape[0]
        self.covmat = (2.4 / np.sqrt(self.n)) ** 2 * s
