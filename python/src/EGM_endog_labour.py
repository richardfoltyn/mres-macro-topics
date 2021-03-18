"""
Topics in Macroeconomics (ECON5098), 2020-21

Solve household problem with endogeous labour income using the endogenous
grid-point method (EGM), given by

    V(a,y) = max_{c,l,a'} { u(c,l) + beta * E[V(a',y')|y] }
        s.t.    c + a' = (1+r)a + w*y*l
                l >= 0, a' >= 0, c >= 0

        with    u(c,l) = (c^(1-gamma)-1)/(1-gamma) - chi*l^(1+1/phi)/(1+1/phi)

Note: This implementation uses the same grid for exogenous savings
    and beginning-of-period assets. This is not required, but spares
    is unnecessary interpolation steps.

Author: Richard Foltyn
"""

import os.path
from time import perf_counter

import numpy as np
from scipy.optimize import root_scalar
from collections import namedtuple

import matplotlib.pyplot as plt

from helpers import powerspace
from env import graphdir
from src.helpers import rouwenhorst, markov_ergodic_dist

GRID_KWARGS = {'b': True, 'lw': 0.5, 'ls': ':', 'alpha': 0.5, 'color': '#333333',
               'zorder': -500}


def main():
    attrs = ['gamma', 'beta', 'chi', 'phi', 'r', 'w', 'lab_ub', 'grid_a',
             'grid_y', 'tm_y']
    Params = namedtuple('Params', attrs)

    # Preference parameters
    gamma = 1.0
    beta = 0.96
    chi = 1.0
    phi = 1.0
    # Optional upper bound on labour supply
    lab_ub = 100.0

    # prices
    r = 0.04
    w = 1.0

    # Parameters for risky labour income process
    rho = 0.95
    sigma = 0.20
    N_y = 3

    # Parameters for asset grid
    a_max = 100.0
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
    par = Params(gamma, beta, chi, phi, r, w, lab_ub, grid_a, grid_y, tm_y)

    # === EGM ===

    # Solve HH problem using EGM
    pfun_cons, pfun_sav, pfun_lab = egm_IH(par)

    # === Plot results ===

    xlim = (0.0, 50.0)
    if xlim[-1] < grid_a[-1]:
        imax = np.where(grid_a > xlim[-1])[0][0] + 1
    else:
        imax = len(grid_a)
    xvalues = grid_a[:imax]

    colours = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3','#ff7f00']
    kw = {'linewidth': 2.0, 'linestyle': '-', 'alpha': 0.7}

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(3.5, 9.5))

    for iy in range(N_y):
        axes[0].plot(xvalues, pfun_sav[iy, :imax], c=colours[iy], **kw)
        axes[1].plot(xvalues, pfun_cons[iy, :imax], c=colours[iy], **kw)
        axes[2].plot(xvalues, pfun_lab[iy, :imax], c=colours[iy], **kw)

    axes[0].set_title(r'Savings $a^{\prime}$')
    axes[0].set_xlabel('Assets')
    axes[0].set_xlim(xlim)
    axes[0].grid(**GRID_KWARGS)

    axes[1].set_title(r'Consumption $c$')
    axes[1].set_xlabel('Assets')
    axes[1].set_xlim(xlim)
    axes[1].grid(**GRID_KWARGS)

    axes[2].set_title(r'Labour supply $\ell$')
    axes[2].set_xlabel('Assets')
    axes[2].set_xlim(xlim)
    axes[2].grid(**GRID_KWARGS)

    fig.tight_layout()
    fn = os.path.join(graphdir, f'EGM_endog_labour_N{N_a}.pdf')
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
    pfun_lab : np.ndarray
        Labour supply policy function
    """

    t0 = perf_counter()

    N_a, N_y = len(par.grid_a), len(par.grid_y)
    shape = (N_y, N_a)

    # Initial guess for consumption policy function, assuming that HH
    # works max. amount
    pfun_cons = (1.0 + par.r) * par.grid_a[None] + par.grid_y[:, None] * par.w
    pfun_cons_upd = np.zeros(shape)

    pfun_lab = np.zeros(shape)
    pfun_lab_upd = np.zeros(shape)

    # Extract parameters from par object
    beta, gamma, chi, phi = par.beta, par.gamma, par.chi, par.phi
    r, w = par.r, par.w

    for it in range(maxiter):

        # Iterate over all labour income states
        for iy, y in enumerate(par.grid_y):

            # Expected marginal utility tomorrow
            mu = np.dot(par.tm_y[iy], pfun_cons**(-gamma))
            # Compute right-hand side of Euler equation (EE)
            ee_rhs = beta * (1.0 + r) * mu

            # Invert EE to get consumption as a function of savings today
            cons_sav = ee_rhs**(-1.0/gamma)

            # Implied labour supply from intratemporal optimality condition
            # Note that lambda = c^-gamma = EE_RHS where lambda is the
            # Lagrange multiplier on the budget constraint.
            labour_sav = (ee_rhs * w * y / chi) ** phi

            # Enforce boundary constraints
            labour_sav = np.fmin(par.lab_ub, labour_sav)

            # Implied earnings
            earn = y * w * labour_sav

            # Use budget constraint to get beginning-of-period assets
            assets_sav = (cons_sav + par.grid_a - earn) / (1.0 + r)

            # Interpolate back onto exogenous savings grid
            # Today's savings become next-period's assets, so this
            # interpolation works.
            pfun_cons_upd[iy] = np.interp(par.grid_a, assets_sav, cons_sav)

            # Interpolate labour policy function
            pfun_lab_upd[iy] = np.interp(par.grid_a, assets_sav, labour_sav)

            # Fix consumption in region where HH does not save
            amin = assets_sav[0]
            idx = np.where(par.grid_a <= amin)[0]

            if len(idx) > 0:
                # Compute implied labour using simplified max. problem
                # max_{c,l} u(c,l) s.t. c = (1+r)a + w*y*l
                for ia, a in enumerate(par.grid_a[idx]):
                    c_opt, l_opt = solve_no_sav(par, a, y)
                    pfun_cons_upd[iy, ia] = c_opt
                    pfun_lab_upd[iy, ia] = l_opt

        # Make sure that labour policy satisfies constraints
        assert np.all(pfun_lab_upd >= 0.0) and np.all(pfun_lab_upd <= par.lab_ub)
        # CAH implied by optimal labour supply choice
        cah = (1.0 + r) * par.grid_a[None] + par.grid_y[:,None] * w * pfun_lab_upd
        # Make sure that consumption policy satisfies constraints
        assert np.all(pfun_cons_upd >= 0.0) and np.all(pfun_cons_upd <= cah)

        # Compute max. absolute difference to policy function from previous
        # iteration.
        diff_c = np.amax(np.abs(pfun_cons - pfun_cons_upd))
        diff_l = np.amax(np.abs(pfun_lab - pfun_lab_upd))
        diff = max(diff_l, diff_c)

        # switch references to policy functions for next iteration
        pfun_cons, pfun_cons_upd = pfun_cons_upd, pfun_cons
        pfun_lab, pfun_lab_upd = pfun_lab_upd, pfun_lab

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

    # CAH implied by optimal labour supply choice
    cah = (1.0 + r) * par.grid_a[None] + par.grid_y[:, None] * w * pfun_lab
    pfun_sav = cah - pfun_cons

    return pfun_cons, pfun_sav, pfun_lab


def solve_no_sav(par, a, y):
    """

    Parameters
    ----------
    par : namedtuple
    a : float
    y : float

    Returns
    -------
    c_opt : float
    l_opt = float
    """

    gamma, chi, phi = par.gamma, par.chi, par.phi
    r, w = par.r, par.w

    def foc(l):
        term1 = -chi * l**(1.0/phi)
        term2 = ((1.0+r)*a + y*w*l)**(-gamma) * w * y
        fx = term1 + term2
        return fx

    # Check boundary solution
    fub = foc(par.lab_ub)

    if fub >= 0.0:
        l_opt = par.lab_ub
    else:
        # Find interior solution
        bracket = [1.0e-10, par.lab_ub - 1.0e-10]
        res = root_scalar(foc, bracket=bracket, method='brentq')
        l_opt = res.root

    # Recover consumption from budget constraint
    c_opt = (1.0 + r) * a + w * y * l_opt
    assert c_opt > 0.0

    return c_opt, l_opt


if __name__ == '__main__':
    main()
