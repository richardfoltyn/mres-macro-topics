"""
Topics in Macroeconomics (ECON5098), 2020-21

Solve household problem with risky labour income using value function
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
from matplotlib.lines import Line2D

from helpers import powerspace
from env import graphdir
from src.helpers import rouwenhorst, markov_ergodic_dist

GRID_KWARGS = {'b': True, 'lw': 0.5, 'ls': ':', 'alpha': 0.5, 'color': '#333333',
               'zorder': -500}


def main():
    Params = namedtuple('Params', ['gamma', 'beta', 'r', 'grid_a', 'grid_y', 'tm_y'])
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

    # Cash-at-hand grid
    cah = (1.0 + par.r) * par.grid_a[None] + par.grid_y[:,None]

    # === Grid search ===
    # solve the HH problem using grid search
    vfun_grid, pfun_isav_grid = vfi_grid(par)
    # Savings policy in levels (not indices)
    pfun_sav_grid = grid_a[pfun_isav_grid]

    # Recover consumption policy function
    pfun_cons_grid = cah - pfun_sav_grid

    # === Linear interpolation ===
    # solve the HH problem using linear interpolation
    vfun_linear, pfun_sav_linear = vfi_interp(par)

    # Recover consumption policy function
    pfun_cons_linear = cah - pfun_sav_linear

    # === Plot results ===
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(9, 3.5))

    xlim = (0.0, 5.0)
    imax = np.where(grid_a > xlim[-1])[0][0] + 1
    xvalues = grid_a[:imax]

    colours = ['#e41a1c', '#377eb8', '#4daf4a']
    kw_grid = {'linewidth': 2.0, 'alpha': 0.55}
    kw_linear = {'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.8}

    for iy in range(N_y):
        colour = colours[iy]
        axes[0].plot(xvalues, vfun_grid[iy,:imax], c=colour, **kw_grid)
        axes[0].plot(xvalues, vfun_linear[iy,:imax], c=colour, **kw_linear)

        axes[1].plot(xvalues, pfun_sav_grid[iy,:imax], c=colour, **kw_grid)
        axes[1].plot(xvalues, pfun_sav_linear[iy,:imax], c=colour, **kw_linear)

        axes[2].plot(xvalues, pfun_cons_grid[iy,:imax], c=colour, **kw_grid)
        axes[2].plot(xvalues, pfun_cons_linear[iy,:imax], c=colour, **kw_linear)

    line_grid = Line2D([], [], color='black', **kw_grid)
    line_linear = Line2D([], [], color='black', **kw_linear)
    axes[0].legend([line_grid, line_linear], ['Grid search', 'Linear'],
                   frameon=False)
    axes[0].set_title('Value func. $V$')
    axes[0].set_xlabel('Assets')
    axes[0].set_xlim(xlim)
    funcs = (vfun_grid, vfun_linear)
    ymin, ymax = axes[0].get_ylim()
    ymax = max(np.ceil(np.amax(f[:,imax])) for f in funcs)
    axes[0].set_ylim((ymin, ymax))
    axes[0].grid(**GRID_KWARGS)

    axes[1].set_title(r'Savings $a^{\prime}$')
    axes[1].set_xlabel('Assets')
    axes[1].set_xlim(xlim)
    funcs = (pfun_sav_grid, pfun_sav_linear)
    ymin, ymax = axes[1].get_ylim()
    ymax = max(np.ceil(np.amax(f[:,imax])) for f in funcs)
    axes[1].set_ylim((ymin, ymax))
    axes[1].grid(**GRID_KWARGS)

    axes[2].set_title(r'Consumption $c$')
    axes[2].set_xlabel('Assets')
    axes[2].set_xlim(xlim)
    funcs = (pfun_cons_grid, pfun_cons_linear)
    ymin, ymax = axes[2].get_ylim()
    ymax = max(np.ceil(np.amax(f[:,imax]) * 2.0) / 2.0 for f in funcs)
    axes[2].set_ylim((ymin, ymax))
    axes[2].grid(**GRID_KWARGS)

    fig.tight_layout()
    fn = os.path.join(graphdir, f'VFI_labour_risk_N{N_a}.pdf')
    fig.savefig(fn)


def vfi_grid(par, tol=1e-5, maxiter=1000):
    """
    Solve the household problem with risky labour income using VFI with grid
    search.

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

    N_a, N_y = len(par.grid_a), len(par.grid_y)
    shape = (N_y, N_a)
    vfun = np.zeros(shape)
    vfun_upd = np.empty(shape)
    # index of optimal savings decision
    pfun_isav = np.empty(shape, dtype=np.uint)

    # pre-compute cash at hand for each (asset, labour) grid point
    cah = (1 + par.r) * par.grid_a[None] + par.grid_y[:,None]

    for it in range(maxiter):

        # Compute expected continuation value E[V(y',a')|y] for each (y,a')
        EV = np.dot(par.tm_y, vfun)

        for iy in range(N_y):
            for ia, a in enumerate(par.grid_a):

                # find all values of a' that are feasible, ie. they satisfy
                # the budget constraint
                ia_to = np.where(par.grid_a <= cah[iy, ia])[0]

                # consumption implied by choice a'
                #   c = (1+r)a + y - a'
                cons = cah[iy, ia] - par.grid_a[ia_to]

                # Evaluate "instantaneous" utility
                if par.gamma == 1.0:
                    u = np.log(cons)
                else:
                    u = (cons**(1.0 - par.gamma) - 1.0) / (1.0 - par.gamma)

                # 'candidate' value for each choice a'
                v_cand = u + par.beta * EV[iy, ia_to]

                # find the 'candidate' a' which maximizes utility
                ia_to_max = np.argmax(v_cand)

                # store results for next iteration
                vopt = v_cand[ia_to_max]
                vfun_upd[iy, ia] = vopt
                pfun_isav[iy, ia] = ia_to_max

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


def vfi_interp(par, tol=1e-5, maxiter=1000):
    """
    Solve the household problem using VFI combined with interpolation
    of the continuation value.

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
    vfun : np.ndarray
        Array containing the value function
    pfun_sav : np.ndarray
        Array containing the savings policy function
    """

    t0 = perf_counter()

    N_a, N_y = len(par.grid_a), len(par.grid_y)
    shape = (N_y, N_a)
    vfun = np.zeros(shape)
    vfun_upd = np.empty(shape)
    # Optimal savings decision
    pfun_sav = np.zeros(shape)

    for it in range(maxiter):

        # Compute expected continuation value E[V(y',a')|y] for each (y,a')
        EV = np.dot(par.tm_y, vfun)

        for iy, y in enumerate(par.grid_y):

            # function to interpolate continuation value
            f_vfun = lambda x: np.interp(x, par.grid_a, EV[iy])

            for ia, a in enumerate(par.grid_a):
                # Solve maximization problem at given asset level
                # Cash-at-hand at current asset level
                cah = (1.0 + par.r) * a + y
                # Restrict maximisation to following interval:
                bounds = (0.0, cah)
                # Arguments to be passed to objective function
                args = (cah, par, f_vfun)
                # perform maximisation
                res = minimize_scalar(f_objective, bracket=bounds, args=args)

                vopt = - res.fun
                sav_opt = float(res.x)

                vfun_upd[iy, ia] = vopt
                pfun_sav[iy, ia] = sav_opt

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

    if sav < 0.0 or sav >= cah:
        return np.inf

    # Consumption implied by savings level
    cons = cah - sav

    # Continuation value interpolated onto asset grid
    vcont = f_vfun(sav)

    # evaluate "instantaneous" utility
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
