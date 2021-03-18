"""
Topics in Macroeconomics (ECON5098), 2020-21

Solve household problem with risky labour income using the endogenous
grid-point method (EGM).

Note: This implementation uses the same grid for exogenous savings
    and beginning-of-period assets. This is not required, but spares
    is unnecessary interpolation steps.

Author: Richard Foltyn
"""

import os.path
from time import perf_counter

import numpy as np
from collections import namedtuple

import matplotlib.pyplot as plt

from helpers import powerspace
from env import graphdir
from src.helpers import rouwenhorst, markov_ergodic_dist

GRID_KWARGS = {'b': True, 'lw': 0.5, 'ls': ':', 'alpha': 0.5, 'color': '#333333',
               'zorder': -500}


def main():
    attrs = ['gamma', 'beta', 'r', 'grid_a', 'grid_y', 'tm_y']
    Params = namedtuple('Params', attrs)
    gamma = 2.0
    beta = 0.96
    r = 0.04

    # Parameters for risky labour income process
    rho = 0.95
    sigma = 0.20
    N_y = 3

    # Parameters for asset grid
    a_max = 50
    N_a = 1000

    # Create asset grid
    grid_a = powerspace(0.0, a_max, N_a, 1.3)

    # Discretised labour income process
    states, tm_y = rouwenhorst(N_y, mu=0.0, rho=rho, sigma=sigma)
    # Ergodic distribution of labour income
    edist = markov_ergodic_dist(tm_y, inverse=True)
    # State space in levels
    grid_y = np.exp(states)
    # Normalise states such that unconditional expectation is 1.0
    grid_y /= np.dot(edist, grid_y)

    # store parameters in common structure
    par = Params(gamma, beta, r, grid_a, grid_y, tm_y)

    # === EGM ===

    # Solve HH problem using EGM
    pfun_cons, pfun_sav, assets_sav, cons_sav = egm_IH(par)

    # === Plot results ===

    xlim = (0.0, 5.0)
    if xlim[-1] < grid_a[-1]:
        imax = np.where(grid_a > xlim[-1])[0][0] + 1
    else:
        imax = len(grid_a)
    xvalues = grid_a[:imax]

    colours = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3','#ff7f00']
    kw = {'linewidth': 2.0, 'linestyle': '-', 'alpha': 0.7}

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(6.5, 3.5))

    for iy in range(N_y):
        axes[0].plot(xvalues, pfun_sav[iy, :imax], c=colours[iy], **kw)
        axes[1].plot(xvalues, pfun_cons[iy, :imax], c=colours[iy], **kw)

    axes[0].set_title(r'Savings $a^{\prime}$')
    axes[0].set_xlabel('Assets')
    axes[0].set_xlim(xlim)
    axes[0].grid(**GRID_KWARGS)

    axes[1].set_title(r'Consumption $c$')
    axes[1].set_xlabel('Assets')
    axes[1].set_xlim(xlim)
    axes[1].grid(**GRID_KWARGS)

    fig.tight_layout()
    fn = os.path.join(graphdir, f'EGM_labour_risk_N{N_a}.pdf')
    fig.savefig(fn)

    # === Plot as functions of exogenous savings grid ===

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(6.5, 3.5))

    imax = np.where(par.grid_a > xlim[-1])[0][0] + 1
    xvalues = par.grid_a[:imax]

    # Create dotted lines as this x-value
    xat = 2.0
    kwline = {'lw': 0.5, 'ls': ':', 'alpha': 0.7, 'color': '#333333',
              'zorder': -500}

    for iy in range(N_y):
        axes[0].plot(xvalues, cons_sav[iy, :imax], c=colours[iy], **kw)
        yat = np.interp(xat, par.grid_a, cons_sav[iy])
        axes[0].axhline(yat, **kwline)

        axes[1].plot(xvalues, assets_sav[iy, :imax], c=colours[iy], **kw)
        yat = np.interp(xat, par.grid_a, assets_sav[iy])
        axes[1].axhline(yat, **kwline)

    axes[0].set_title(r'Consumption $c$')
    axes[0].set_xlabel(r'Exogenous savings $a^{\prime}$')
    axes[0].set_xlim(xlim)
    axes[0].axvline(xat, **kwline)

    axes[1].set_title(r'Assets $a$')
    axes[1].set_xlabel(r'Exogenous savings $a^{\prime}$')
    axes[1].set_xlim(xlim)
    axes[1].axvline(xat, **kwline)

    fig.tight_layout()
    fn = os.path.join(graphdir, f'EGM_labour_risk_exog_N{N_a}.pdf')
    fig.savefig(fn)

    # === Compare to solution with purely transitory shocks ===

    # Change transition matrix to be identical to the ergodic distr.
    tm_y = np.tile(edist[None], reps=(N_y, 1))

    # store parameters in common structure
    par = Params(gamma, beta, r, grid_a, grid_y, tm_y)

    # Solve HH problem using EGM
    pfun_cons2, pfun_sav2, *rest = egm_IH(par)

    # === Plot results ===

    xlim = (0.0, 5.0)
    if xlim[-1] < grid_a[-1]:
        imax = np.where(grid_a > xlim[-1])[0][0] + 1
    else:
        imax = len(grid_a)
    xvalues = grid_a[:imax]

    colours = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3','#ff7f00']
    kw = {'linewidth': 1.5, 'linestyle': '-', 'alpha': 0.7}
    kw2 = {'linewidth': 2.0, 'linestyle': '--', 'alpha': 0.8}

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(6.5, 3.5))

    for iy in range(N_y):
        axes[0].plot(xvalues, pfun_sav[iy, :imax], c=colours[iy], **kw)
        axes[1].plot(xvalues, pfun_cons[iy, :imax], c=colours[iy], **kw)

        # Policy functions with transitory shocks
        axes[0].plot(xvalues, pfun_sav2[iy, :imax], c=colours[iy], **kw2)
        axes[1].plot(xvalues, pfun_cons2[iy, :imax], c=colours[iy], **kw2)

    axes[0].set_title(r'Savings $a^{\prime}$')
    axes[0].set_xlabel('Assets')
    axes[0].set_xlim(xlim)
    axes[0].grid(**GRID_KWARGS)

    axes[1].set_title(r'Consumption $c$')
    axes[1].set_xlabel('Assets')
    axes[1].set_xlim(xlim)
    axes[1].grid(**GRID_KWARGS)

    fig.tight_layout()
    fn = os.path.join(graphdir, f'EGM_labour_trans_risk_N{N_a}.pdf')
    fig.savefig(fn)



def egm_IH(par, tol=1.0e-8, maxiter=10000):
    """
    Solve infinite-horizon problem using EGM.

    Parameters
    ----------
    par : namedtuple
        Model parameters
    tol : float, optional
        Termination tolerance
    maxiter : int, optional
        Max. number of iterations

    Returns
    -------
    pfun_cons : np.ndarray
        Consumption policy function defined on beginning-of-period asset grid.
    pfun_sav : np.ndarray
        Savings policy function defined on beginning-of-period asset grid.
    assets_sav : np.ndarray
        Implied beginning-of-period assets defined on exogenous savings grid.
    cons_sav : np.ndarray
        Optimal consumption defined on exogenous savings grid.
    """

    t0 = perf_counter()

    N_a, N_y = len(par.grid_a), len(par.grid_y)
    shape = (N_y, N_a)

    # Cash-at-hand at every asset/savings level
    cah = (1.0 + par.r) * par.grid_a[None] + par.grid_y[:, None]

    # Initial guess for consumption policy function
    pfun_cons = np.copy(cah)
    pfun_cons_upd = np.zeros(shape)

    # Consumption defined on exogenous savings grid
    cons_sav = np.zeros(shape)
    # Inverse savings policy function mapping savings into
    # beginning-of-period assets
    assets_sav = np.zeros(shape)

    # Extract parameters from par object
    beta, gamma, r = par.beta, par.gamma, par.r

    for it in range(maxiter):

        # Iterate over all labour income states
        for iy, y in enumerate(par.grid_y):

            # Expected marginal utility tomorrow
            mu = np.dot(par.tm_y[iy], pfun_cons**(-gamma))
            # Compute right-hand side of Euler equation (EE)
            ee_rhs = beta * (1.0 + r) * mu

            # Invert EE to get consumption as a function of savings today
            cons_sav[iy] = ee_rhs**(-1.0/gamma)

            # Use budget constraint to get beginning-of-period assets
            assets_sav[iy] = (cons_sav[iy] + par.grid_a - y) / (1.0 + r)

            # Interpolate back onto exogenous savings grid
            # Today's savings become next-period's assets, so this
            # interpolation works.
            pfun_cons_upd[iy] = np.interp(par.grid_a, assets_sav[iy], cons_sav[iy])

            # Fix consumption in region where HH does not save
            amin = assets_sav[iy, 0]
            idx = np.where(par.grid_a <= amin)[0]
            # HH consumes entire cash-at-hand
            pfun_cons_upd[iy, idx] = cah[iy, idx]

        # Make sure that consumption policy satisfies constraints
        assert np.all(pfun_cons_upd >= 0.0) and np.all(pfun_cons_upd <= cah)

        # Compute max. absolute difference to policy function from previous
        # iteration.
        diff = np.max(np.abs(pfun_cons - pfun_cons_upd))

        # switch references to policy functions for next iteration
        pfun_cons, pfun_cons_upd = pfun_cons_upd, pfun_cons

        if diff < tol:
            # Convergence achieved, exit loop
            td = perf_counter() - t0
            msg = f'EGM: Converged after {it:d} iterations ({td:.1f} sec.): ' \
                  f'd(c)={diff:4.2e}'
            print(msg)
            break
        elif it == 1 or it % 10 == 0:
            msg = f'EGM: Iteration {it:4d}, dV={diff:4.2e}'
            print(msg)
    else:
        msg = f'Did not converge in {it:d} iterations'
        print(msg)

    pfun_sav = cah - pfun_cons

    return pfun_cons, pfun_sav, assets_sav, cons_sav


if __name__ == '__main__':
    main()
