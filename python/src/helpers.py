
import numpy as np


def powerspace(xmin, xmax, n, exponent):
    """
    Create a "power-spaced" grid of size n.

    Parameters
    ----------
    xmin : float
        Lower bound
    xmax : float
        Upper bound
    n : int
        Number of grid points
    exponent : float
        Shape parameter of "power-spaced" grid.

    Returns
    -------
    xx : np.ndarray
        Array containing "power-spaced" grid
    """

    N = int(n)
    ffrom, fto = float(xmin), float(xmax)
    fexponent = float(exponent)

    zz = np.linspace(0.0, 1.0, N)
    if fto > ffrom:
        xx = ffrom + (fto - ffrom) * zz**fexponent
        # Prevent rounding errors
        xx[-1] = fto
    else:
        xx = ffrom - (ffrom - fto) * zz**fexponent
        xx[0] = ffrom
        xx = xx[::-1]

    return xx


def rouwenhorst(n, mu, rho, sigma):
    """
    Code to approximate an AR(1) process using the Rouwenhorst method as in
    Kopecky & Suen, Review of Economic Dynamics (2010), Vol 13, pp. 701-714
    Adapted from Matlab code by Martin Floden.

    Parameters
    ----------
    n : int
        Number of states for discretized Markov process
    mu : float
        Unconditional mean or AR(1) process
    rho : float
        Autocorrelation of AR(1) process
    sigma : float
        Conditional standard deviation of AR(1) innovations

    Returns
    -------
    z : numpy.ndarray
        Discretized state space
    Pi : numpy.ndarray
        Transition matrix of discretized process where
            Pi[i,j] = Prob[z'=z_j | z=z_i]
    """

    if n < 1:
        msg = 'Invalid number of states'
        raise ValueError(msg)
    if sigma < 0.0:
        msg = 'Argument sigma must be non-negative'
        raise ValueError(msg)
    if abs(rho) >= 1.0:
        msg = 'Cannot create stationary process with abs(rho) >= 1.0'
        raise ValueError(msg)

    if n == 1:
        # Degenerate process on a single state: disregard variance and
        # autocorrelation
        z = np.array([mu])
        Pi = np.ones((1, 1))
        return z, Pi

    p = (1+rho)/2
    Pi = np.array([[p, 1-p], [1-p, p]])

    for i in range(Pi.shape[0], n):
        tmp = np.pad(Pi, 1, mode='constant', constant_values=0)
        Pi = p * tmp[1:, 1:] + (1-p) * tmp[1:, :-1] + \
             (1-p) * tmp[:-1, 1:] + p * tmp[:-1, :-1]
        Pi[1:-1, :] /= 2

    fi = np.sqrt(n-1) * sigma / np.sqrt(1 - rho ** 2)
    z = np.linspace(-fi, fi, n) + mu

    return z, Pi


def markov_ergodic_dist(transm, tol=1e-12, maxiter=10000, transpose=True,
                        mu0=None, inverse=True):
    """
    Compute the ergodic distribution implied by a given Markov chain transition
    matrix.

    Parameters
    ----------
    transm : numpy.ndarray
        Markov chain transition matrix
    tol : float
        Terminal tolerance on consecutive changes in the ergodic distribution
        if computing via the iterative method (`inverse` = False)
    maxiter : int
        Maximum number of iterations for iterative method.
    transpose : bool
        If true, the transition matrix `transm` is provided in transposed form.
    mu0 : numpy.ndarray
        Optional initial guess for the ergodic distribution if the iterative
        method is used (default: uniform distribution).
    inverse : bool
        If True, compute the erdogic distribution using the "inverse" method

    Returns
    -------
    mu : numpy.ndarray
        Ergodic distribution
    """

    # This function should also work for sparse matrices from scipy.sparse,
    # so do not use .T to get the transpose.
    if transpose:
        transm = transm.transpose()

    assert np.all(np.abs(transm.sum(axis=0) - 1) < 1e-12)

    if not inverse:
        if mu0 is None:
            # start out with uniform distribution
            mu0 = np.ones((transm.shape[0], ), dtype=np.float64)/transm.shape[0]

        for it in range(maxiter):
            mu1 = transm.dot(mu0)

            dv = np.max(np.abs(mu0 - mu1))
            if dv < tol:
                mu = mu1 / np.sum(mu1)
                break
            else:
                mu0 = mu1
        else:
            msg = 'Failed to converge (delta = {:.e})'.format(dv)
            print(msg)
            raise RuntimeError(it, dv)
    else:
        m = transm - np.identity(transm.shape[0])
        m[-1] = 1
        m = np.linalg.inv(m)
        mu = np.ascontiguousarray(m[:, -1])
        assert np.abs(np.sum(mu) - 1) < 1e-9
        mu /= np.sum(mu)

    return mu


def interp1d(x, xp, fp, extrapolate=True):
    """
    Wrapper for NumPy's extrapolation

    Parameters
    ----------
    x
    xp
    fp
    extrapolate

    Returns
    -------

    """
    if np.isscalar(x):
        if extrapolate:
            if x < xp[0]:
                xlb, xub = xp[0], xp[1]
                wgt = (xub - x) / (xub - xlb)
                fx = wgt * fp[0] + (1.0 - wgt) * fp[1]
                return fx
            elif x > xp[-1]:
                xlb, xub = xp[-2], xp[-1]
                wgt = (xub - x) / (xub - xlb)
                fx = wgt * fp[-2] + (1.0 - wgt) * fp[-1]
                return fx
            else:
                fx = np.interp(x, xp, fp)
                return fx
        else:
            fx = np.interp(x, xp, fp)
            return fx
    else:
        out = np.interp(x, xp, fp)

        if extrapolate:
            ii = np.where(x < xp[0])[0]
            if len(ii) > 0:
                xlb, xub = xp[0], xp[1]
                wgt = (xub - x[ii]) / (xub - xlb)
                fx = wgt * fp[0] + (1.0 - wgt) * fp[1]
                out[ii] = fx

            ii = np.where(x > xp[-1])[0]
            if len(ii) > 0:
                xlb, xub = xp[-2], xp[-1]
                wgt = (xub - x[ii]) / (xub - xlb)
                fx = wgt * fp[-2] + (1.0 - wgt) * fp[-1]
                out[ii] = fx

        return out


try:
    from numba import jit


    @jit(inline='always', nopython=True, nogil=True, parallel=False)
    def bsearch(needle, haystack, ilb=0):
        """

        Parameters
        ----------
        needle :
        haystack : np.ndarray
        ilb : int
            Cached value of index of lower bound of bracketing interval.

        Returns
        -------
        ilb : int
            Index of lower bound of bracketing interval.
        """

        n = haystack.shape[0]
        iub = n - 1

        if haystack[ilb] <= needle:
            if haystack[ilb + 1] > needle:
                return ilb
            elif ilb == (n - 2):
                return ilb
        else:
            ilb, iub = 0, ilb

        while iub > (ilb + 1):
            imid = (iub + ilb) // 2
            if haystack[imid] > needle:
                iub = imid
            else:
                ilb = imid

        return ilb


    @jit(nopython=True, nogil=True, parallel=False)
    def interp1d(x, xp, fp, extrapolate=True):

        xarr = np.asarray(x)
        x1d = np.atleast_1d(xarr)
        res1d = np.empty_like(x1d)

        if xarr.ndim == 0:
            ilb = bsearch(x, xp)
            xlb, xub = xp[ilb], xp[ilb + 1]
            wgt = (xub - x) / (xub - xlb)

            if not extrapolate:
                wgt = max(0.0, min(1.0, wgt))

            fxi = wgt * fp[ilb] + (1.0 - wgt) * fp[ilb + 1]
            res1d[0] = fxi
        else:
            ilb = 0

            for i, xi in enumerate(x1d):
                ilb = bsearch(xi, xp, ilb)
                xlb, xub = xp[ilb], xp[ilb+1]
                wgt = (xub - xi) / (xub - xlb)

                if not extrapolate:
                    wgt = max(0.0, min(1.0, wgt))

                fxi = wgt * fp[ilb] + (1.0 - wgt) * fp[ilb+1]
                res1d[i] = fxi

        return res1d


except ImportError:
    pass



