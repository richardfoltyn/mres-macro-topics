
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
        Markov chain transition matrix where the element at position (i,j)
        represents the transition probability from i to j.
    tol : float
        Terminal tolerance on consecutive changes in the ergodic distribution
        if computing via the iterative method (`inverse` = False)
    maxiter : int
        Maximum number of iterations for iterative method.
    transpose : bool
        If false, the transition matrix `transm` is assumed to be in transposed
        form, i.e. each element (i,j) represents the transition probability
        from state j to state i.
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

