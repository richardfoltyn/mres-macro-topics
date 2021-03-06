"""
Topics in Macroeconomics (ECON5098), 2020-21

Solve household problem with determinsitic labour income using value function
iteration (VFI).

This file implements two different solution methods:
    1.  VFI with grid search, allowing only for savings choices on the
        asset grid.
    2.  VFI with interpolation, allowing for savings choices that need not
        be on the asset grid.


Author: Richard Foltyn
"""

import os.path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from collections import namedtuple

import matplotlib.pyplot as plt

from helpers import powerspace
from env import graphdir


GRID_KWARGS = {'b': True, 'lw': 0.5, 'ls': ':', 'alpha': 0.5, 'color': '#333333',
               'zorder': -500}


def main():
    Params = namedtuple('Params', ['gamma', 'beta', 'r', 'y', 'grid_a'])
    gamma = 2.0
    beta = 0.96
    r = 0.04

    # Parameters for asset grid
    a_max = 50
    N_a = 50

    # Create asset grid
    grid_a = powerspace(0.0, a_max, N_a, 1.3)

    # store parameters in common structure
    y = 1.0
    par = Params(gamma, beta, r, y, grid_a)

    # Cash-at-hand for each asset grid point
    cah = (1.0 + par.r) * grid_a + par.y

    # === Grid search ===
    # solve the HH problem using grid search
    vfun_grid, pfun_isav_grid = vfi_grid(par)
    # Savings policy in levels (not indices)
    pfun_sav_grid = grid_a[pfun_isav_grid]

    # Recover consumption policy function
    pfun_cons_grid = cah - pfun_sav_grid

    # === Linear interpolation ===
    # solve the HH problem using linear interpolation
    vfun_linear, pfun_sav_linear = vfi_interp(par, kind='linear')

    # Recover consumption policy function
    pfun_cons_linear = cah - pfun_sav_linear

    # === Spline interpolation ===
    vfun_cubic, pfun_sav_cubic = vfi_interp(par, kind='cubic')

    # Recover consumption policy function
    pfun_cons_cubic = cah - pfun_sav_cubic

    # === Plot results ===

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(9, 3.5))

    xlim = (0.0, 5.0)
    imax = np.where(grid_a > xlim[-1])[0][0] + 1
    xvalues = grid_a[:imax]

    axes[0].plot(xvalues, vfun_grid[:imax], lw=2.0, alpha=0.7, label='Grid search')
    axes[0].plot(xvalues, vfun_linear[:imax], lw=1.5, alpha=0.7, c='darkred',
                 ls='--', label='Linear')
    axes[0].plot(xvalues, vfun_cubic[:imax], lw=1.0, alpha=0.7, c='black',
                 ls=':', label='Cubic')
    axes[0].legend(loc='upper left', frameon=False)
    axes[0].set_title('Value func. $V$')
    axes[0].set_xlabel('Assets')
    axes[0].set_xlim(xlim)
    funcs = (vfun_grid, vfun_linear, vfun_cubic)
    ymin, ymax = axes[0].get_ylim()
    ymax = max(np.ceil(f[imax]) for f in funcs)
    axes[0].set_ylim((ymin, ymax))
    axes[0].grid(**GRID_KWARGS)

    axes[1].plot(xvalues, pfun_sav_grid[:imax], lw=2.0, alpha=0.7)
    axes[1].plot(xvalues, pfun_sav_linear[:imax], lw=1.5, alpha=0.7, c='darkred', ls='--')
    axes[1].plot(xvalues, pfun_sav_cubic[:imax], lw=1.0, alpha=0.7, c='black', ls=':')
    axes[1].set_title(r'Savings $a^{\prime}$')
    axes[1].set_xlabel('Assets')
    axes[1].set_xlim(xlim)
    funcs = (pfun_sav_grid, pfun_sav_linear, pfun_sav_cubic)
    ymin, ymax = axes[1].get_ylim()
    ymax = max(np.ceil(f[imax]) for f in funcs)
    axes[1].set_ylim((ymin, ymax))
    axes[1].grid(**GRID_KWARGS)

    axes[2].plot(xvalues, pfun_cons_grid[:imax], lw=2.0, alpha=0.7)
    axes[2].plot(xvalues, pfun_cons_linear[:imax], lw=1.5, alpha=0.7, c='darkred', ls='--')
    axes[2].plot(xvalues, pfun_cons_cubic[:imax], lw=1.0, alpha=0.7, c='black', ls=':')
    axes[2].set_title(r'Consumption $c$')
    axes[2].set_xlabel('Assets')
    axes[2].set_xlim(xlim)
    funcs = (pfun_cons_grid, pfun_cons_linear, pfun_cons_cubic)
    ymin, ymax = axes[2].get_ylim()
    ymax = max(np.ceil(f[imax] * 10.0) / 10.0 for f in funcs)
    axes[2].set_ylim((ymin, ymax))
    axes[2].grid(**GRID_KWARGS)

    fig.tight_layout()
    fn = os.path.join(graphdir, f'VFI_labour_N{N_a}.pdf')
    fig.savefig(fn)


def vfi_grid(par, tol=1e-5, maxiter=1000):
    """
    Solve the household problem using VFI with grid search.

    Parameters
    ----------
    par : namedtuple
        Model parameters and grids
    tol : float, optional
        Termination tolerance
    maxiter : int, optional
        Max. number of iterations

    Returns
    -------
    vfun : np.ndarray
        Array containing the value function
    pfun_sav : np.ndarray
        Array containing the savings policy function
    """

    t0 = perf_counter()

    N_a = len(par.grid_a)
    vfun = np.zeros(N_a)
    vfun_upd = np.empty(N_a)
    # index of optimal savings decision
    pfun_isav = np.empty(N_a, dtype=np.uint)

    # pre-compute cash at hand for each asset grid point
    cah = (1 + par.r) * par.grid_a + par.y

    for it in range(maxiter):

        for ia, a in enumerate(par.grid_a):

            # find all values of a' that are feasible, ie. they satisfy
            # the budget constraint
            ia_to = np.where(par.grid_a <= cah[ia])[0]

            # consumption implied by choice a'
            #   c = (1+r)a + y - a'
            cons = cah[ia] - par.grid_a[ia_to]

            # Evaluate "instantaneous" utility
            if par.gamma == 1.0:
                u = np.log(cons)
            else:
                u = (cons**(1.0 - par.gamma) - 1.0) / (1.0 - par.gamma)

            # 'candidate' value for each choice a'
            v_cand = u + par.beta * vfun[ia_to]

            # find the 'candidate' a' which maximizes utility
            ia_to_max = np.argmax(v_cand)

            # store results for next iteration
            vopt = v_cand[ia_to_max]
            vfun_upd[ia] = vopt
            pfun_isav[ia] = ia_to_max

        diff = np.max(np.abs(vfun - vfun_upd))

        # switch references to value functions for next iteration
        vfun, vfun_upd = vfun_upd, vfun

        if diff < tol:
            td = perf_counter() - t0
            msg = f'VFI: Converged after {it:3d} iterations ({td:.1f} sec.): dV={diff:4.2e}'
            print(msg)
            break
        elif it == 1 or it % 10 == 0:
            msg = f'VFI: Iteration {it:3d}, dV={diff:4.2e}'
            print(msg)
    else:
        msg = f'Did not converge in {it:d} iterations'
        print(msg)

    return vfun, pfun_isav


def vfi_interp(par, kind='linear', tol=1e-5, maxiter=1000):
    """
    Solve the household problem using VFI combined with interpolation
    of the continuation value.

    Parameters
    ----------
    par : namedtuple
        Model parameters
    kind : str, optional
        Type of interpolation to perform on the continuation value.
    tol : float, optional
        Termination tolerance
    maxiter : int, optional
        Max. number of iterations

    Returns
    -------
    vfun : np.ndarray
        Array containing the value function
    pfun_sav : np.ndarray
        Array containing the savings policy function
    """

    t0 = perf_counter()

    N_a = len(par.grid_a)
    vfun = np.zeros(N_a)
    vfun_upd = np.empty(N_a)
    # Optimal savings decision
    pfun_sav = np.zeros(N_a)

    for it in range(maxiter):

        if kind == 'linear':
            f_vfun = lambda x: np.interp(x, par.grid_a, vfun)
        else:
            f_vfun = interp1d(par.grid_a, vfun, kind=kind, bounds_error=False,
                              fill_value='extrapolate', assume_sorted=True,
                              copy=False)

        for ia, a in enumerate(par.grid_a):
            # Solve maximization problem at given asset level
            # Cash-at-hand at current asset level
            cah = (1.0 + par.r) * a + par.y
            # Restrict maximisation to following interval:
            bounds = (0.0, cah)
            # Arguments to be passed to objective function
            args = (cah, par, f_vfun)
            # perform maximisation
            res = minimize_scalar(f_objective, bracket=bounds, args=args)

            vopt = - res.fun
            sav_opt = float(res.x)

            vfun_upd[ia] = vopt
            pfun_sav[ia] = sav_opt

        diff = np.max(np.abs(vfun - vfun_upd))

        # switch references to value functions for next iteration
        vfun, vfun_upd = vfun_upd, vfun

        if diff < tol:
            td = perf_counter() - t0
            msg = f'VFI: Converged after {it:3d} iterations ({td:.1f} sec.): dV={diff:4.2e}'
            print(msg)
            break
        elif it == 1 or it % 10 == 0:
            msg = f'VFI: Iteration {it:3d}, dV={diff:4.2e}'
            print(msg)
    else:
        msg = f'Did not converge in {it:d} iterations'
        print(msg)

    return vfun, pfun_sav


def f_objective(sav, cah, par, f_vfun):
    """
    Objective function for the minimizer.

    Parameters
    ----------
    sav : float
        Current guess for optional savings
    cah : float
        Current CAH level
    par : namedtuple
        Model parameters
    f_vfun : callable
        Function interpolating the continuation value.

    Returns
    -------
    float
        Objective function evaluated at given savings level
    """

    sav = float(sav)
    if sav < 0.0 or sav > cah:
        return np.inf

    # Consumption implied by savings level
    cons = cah - sav

    # Continuation value interpolated onto asset grid
    vcont = f_vfun(sav)

    # current-period utility
    if cons <= 0.0:
        u = - np.inf
    else:
        if par.gamma == 1.0:
            u = np.log(cons)
        else:
            u = (cons**(1.0 - par.gamma) - 1.0) / (1.0 - par.gamma)

    # Objective evaluated at current savings level
    obj = u + par.beta * vcont

    # We are running a minimiser, return negative of objective value
    return -obj


if __name__ == '__main__':
    main()
