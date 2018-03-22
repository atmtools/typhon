"""
The mcmc submodule.
===================

Contains the MCMC which implements the Marcov Chain Monte Carlo method to
sample from a posterior distribution of a retrieval problem.

References
==========

[1] Andrew Gelman et al., Bayesian Data Analysis, 3rd Edition

"""
import ctypes as c
import numpy as np
import matplotlib as plt

from typhon.arts.workspace import Workspace

def r_factor(stats):
    """
    This computes the R-factor as defined in 'Bayesian Data Analysis'
    (Getlman et al.), Chapter 11. If the simulations have converged,
    the result of `r_factor` should be close to one.

    Args:
        stats: A list of arrays of statistics (scalar summaries) computed from
        serveral MCMC runs.
    """
    n = stats[0].size
    m = len(stats)

    vars  = np.array([np.var(s) for s in stats])
    means = np.array([np.mean(s) for s in stats])
    mean  = np.mean(means)

    b = n * np.var(means)
    w = np.mean(vars)

    var_p = (n - 1) / n * w + b / n
    return np.sqrt(var_p / w)

def variogram(stats, t):
    """
    Helper function that computes the variogram for a given lag t. The variogram
    is the mean of the mean squared sum of deviations of lag t of each sequence.

    Args:
        stats: A list of sequences
    """
    m = len(stats)
    n = stats[0].size
    return sum([np.sum((s[t+1:] - s[:n-t-1])**2) for s in stats]) / (m * (n - t))

def split(stats):
    """
    Splits a list of sequences in halves.

    Sequences generated from MCMC runs should be split in half in order to be able to
    properly diagnose mixing.

    Args:
        stats: A list of sequences
    """
    n = stats[0].size
    return [s[i * (n // 2) : (i + 1) * (n // 2)]
            for i in range(2) for s in stats]

def autocorrelation(stats):
    """
    Estimates the autocorrelation of a list of sequences from a MCMC run.
    This uses formula (11.7) in [1] to approximate the autocorrelation function
    for lags [0, n // 2].
    """
    n = stats[0].size
    m = len(stats)

    vars  = np.array([np.var(s) for s in stats])
    means = np.array([np.mean(s) for s in stats])
    mean  = np.mean(means)
    b = n * np.var(means)
    w = np.mean(vars)
    var_p = (n - 1) / n * w + b / n

    rho_tp1 = 0.0
    rho_tp2 = 0.0
    rho = np.zeros(n // 2)
    for t in range(n // 2):
        vt     = variogram(stats, t)
        rho[t] = 1.0 - 0.5 * vt / var_p
    return rho

def effective_sample_size(stats):
    """
    This estimates the effective sample size of independent samples from the
    posterior distribution using formula (11.8) in [1].
    """
    n = stats[0].size
    m = len(stats)

    vars  = np.array([np.var(s) for s in stats])
    means = np.array([np.mean(s) for s in stats])
    mean  = np.mean(means)
    b = n * np.var(means)
    w = np.mean(vars)
    var_p = (n - 1) / n * w + b / n

    rho = np.zeros(n // 2)
    for t in range(n // 2):
        vt     = variogram(stats, t)
        rho[t] = 1.0 - 0.5 * vt / var_p
        if t > 2 and (rho[t-1] + rho[t-2]) < 0.0:
            break

    return m * n / (1.0 + 2.0 * sum(rho[:t-2]))

class MCMC:
    """
    The MCMC class represents an ongoing MCMC simulation. An MCMC object can be
    used to run a given number of MC steps, test the results for convergence
    and perform further calculations if necessary.
    """
    @staticmethod
    def _check_input(vars, py, stats):
        """
        Helper method that checks arguments provided to `__init__`.
        """
        if not type(vars) == list and len(vars) > 0:
            raise Exception("Argument vars must be of type list and have length"
                            + "> 0.")
        for v in vars:
            if not type(v) == list and len(v) == 3:
                raise Exception("Elements of argument vars must be tuples of "
                                + " mutable variables, jump functions and prior"
                                + " densities.")
            if not callable(v[1]):
                raise Exception("Non-callable object given as prior density.")
            if not callable(v[2]):
                raise Exception("Non-callable object given as jump function.")
        if not callable(py):
                raise Exception("Non-callable object given for conditional"
                                + "probability p(y | x).")
        for s in stats:
            if not callable(s):
                raise Exception("Non-callable object given as statistic.")


    def __init__(self, vars, y, ly, stats = []):
        """
        To construct an MCMC object, the user must provide a list of variables,
        prior distributions and likelihood functions, the measurement vector,
        a measurement likelihood and optionally a set of stats to evaluate at
        each step.

        Args:
           vars: A list of triples (v,l,j) containing a triple of a variable
           v, a prior likelihood function l so that `l(v)` yields a value
           proportional to the logarithm of the prior probability of value
           of v, and finally a jump function j, so that `v_new = j(ws, v_old)` 
           yields a new value for the variable v and manipulates the 
           `Workspace` object ws so that a subsequent call to the yCalc WSM
           will compute the simulated measurement corresponding to the
           new value `v_new` of the variable v.

           y: The measured vector of brightness temperatures which must be
           consistent with the ARTS WSV y

           ly: The measurement likelihood such that `ly(y, yf)` gives
           the log of the probability that deviations between `y` and
           `yf` are due to measurement errors.

           stats: This is a list of statstics such that for each element
           s `s(ws)` is a scalar value computed on a given workspace.
        """
        MCMC._check_input(vars, ly, stats)
        self.y     = y
        self.vars  = vars
        self.ly    = ly
        self.stats = stats

    def eval_l(self, ws):
        """
        Evaluate the likelihood of the current state. This method
        simply computes and sums up the measurement likelihood and
        the prior likelihoods.

        Args:
           ws: A `Workspace` object consistent with the current state
           of the MCMC run.
        """
        lxs = np.zeros(len(self.vars))
        ly  = self.ly(self.y, ws.y.value)
        for i, (x, l, _) in enumerate(self.vars):
            lxs[i] = l(x)
        return ly, lxs

    def step(self, ws, ly_old, lxs_old):
        """
        The performs a Gibbs step for a given variable. This will generate
        a candidate for the given variable using the corresponding jump
        function, call `yCalc()` on the `Workspace` object `ws`

        Args:
           ws: A `Workspace` object consistent with the current state
           of the MCMC run.
           ly_old: The measurement likelihood before the execution of
           new step
           lxs_old: The prior likelihoods for each of the variables
           that are being retrieved.
        """
        accepted = np.zeros((1, len(self.vars)), dtype=bool)
        lxs = np.zeros(lxs_old.shape)
        for i, ((x,l,j), lx_old) in enumerate(zip(self.vars, lxs_old)):

            # Generate new step
            x_new = j(ws, x)
            ws.yCalc()
            lx_new = l(x_new)
            ly_new = self.ly(self.y, ws.y.value)

            # Check Acceptance
            r = np.exp(lx_new + ly_new - lx_old - ly_old)
            if r > 1.0 or np.random.random() < r:
                x[:] = x_new
                accepted[0, i] = True
                ly_old = ly_new
                lxs[i] = lx_new
            else:
                j(ws, x, revert = True)
                accepted[0, i] = False
                lxs[i] = lxs_old[i]

        return accepted, ly_old, lxs

    def print_log(self, step, acceptance):
        """
        Prints log output to stdout.

        Args:
            step: The number of the current step
            acceptance: The array of bools tracking the acceptances for
            each simulation step.
        """
        if step > 0:
            ar = sum(acceptance / step)
        else:
            ar = 0.0

        print("MCMC Step " + str(step) + ": " + "ar = " + str(ar))

    def warm_up(self, ws, x0s, n_steps):
        """
        Run a simulation of `n_steps` on a given workspace `ws`
        starting from start values `x0s`.

        Args:
            ws: A `Workspace` object setup so that only a call to the `yCalc`
            WSM is necessary to perform a simulation
            x0s: A list of start values which is used to initialized the
            workspace by calling `j(x0)` for each `x0` in `x0s` and `j` is
            the jump function of the corresponding variable.
        """
        ls    = np.zeros(n_steps + 1)
        stats = np.zeros((n_steps + 1, len(self.stats)))
        hist  = [np.zeros((n_steps + 1,) + x.shape) for x,l,j in self.vars]
        acceptance = np.zeros((n_steps, len(self.vars)), dtype=bool)

        lxs = np.zeros(len(self.vars))
        ly  = 0.0

        # Set initial state.
        for i,((x, l, j), x0) in enumerate(zip(self.vars, x0s)):
            x[:] = j(ws, x0)
            hist[i][0,:] = x[:]

        # Evaluate likelihood
        ws.yCalc()
        ly, lxs = self.eval_l(ws)
        ls[0] = ly + sum(lxs)

        # Evaluate statistics
        for i,s in enumerate(self.stats):
            stats[0, i] = s(ws)

        for i1 in range(n_steps):
            acceptance[i1, :], ly, lxs = self.step(ws, ly, lxs)
            for i2, h in enumerate(hist):
                hist[i2][i1+1, :] = self.vars[i2][0][:]
            for i2,s in enumerate(self.stats):
                    stats[i1 + 1, i2] = s(ws)
            if (i1 % 10) == 0:
                self.print_log(i1, acceptance)
        self.ly_old    = ly
        self.lxs_old   = lxs
        self.stats_old = stats[-1,:]
        self.hist_old  = [h[-1,:] for h in hist]
        return hist, stats, ls, acceptance

    def run(self, ws, n_steps):
        """
        Run a simulation of `n_steps` on a given workspace `ws`
        starting from start values `x0s`.

        Args:
            ws: A `Workspace` object setup so that only a call to the `yCalc`
            WSM is necessary to perform a simulation
            x0s: A list of start values which is used to initialized the
            workspace by calling `j(x0)` for each `x0` in `x0s` and `j` is
            the jump function of the corresponding variable.
        """
        ls    = np.zeros(n_steps)
        stats = np.zeros((n_steps, len(self.stats)))
        hist  = [np.zeros((n_steps,) + x.shape) for x,l,j in self.vars]
        acceptance = np.zeros((n_steps, len(self.vars)), dtype=bool)

        ly  = self.ly_old
        lxs = self.lxs_old

        for i1 in range(n_steps):
            acceptance[i1, :], ly, lxs = self.step(ws, ly, lxs)
            for i2, h in enumerate(hist):
                hist[i2][i1, :] = self.vars[i2][0][:]
            for i2,s in enumerate(self.stats):
                    stats[i1, i2] = s(ws)
            if (i1 % 10) == 0:
                self.print_log(i1, acceptance)
        self.ly_old  = ly
        self.lxs_old = lxs
        self.stats_old = stats[-1,:]
        self.hist_old  = [h[-1,:] for h in hist]
        return hist, stats, ls, acceptance
