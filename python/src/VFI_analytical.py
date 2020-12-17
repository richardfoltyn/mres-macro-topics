"""
Code to solve VFI for log preferences and no labor income. In this case
we can derive the analytical solution to compare to our numerical results.

Author: Richard Foltyn
"""

import os.path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from collections import namedtuple

from helpers import powerspace
from env import graphdir

import matplotlib.pyplot as plt


def main():
    Params = namedtuple('Params', ['beta', 'r', 'grid_a'])
    beta = 0.96
    r = 0.04

    # Parameters for asset grid
    a_max = 5
    N_a = 100

    # Create asset grid
    grid_a = powerspace(1.0e-8, a_max, N_a, 1.5)

    # store parameters in common structure
    par = Params(beta, r, grid_a)

    # Solve for sequence of coefficients using analytical solution and
    # plot result
    fn = os.path.join(graphdir, 'VFI_analytical_coefs.pdf')
    plot_coefs_vfun(par, fn)

    # Solve for sequence of MPCs using analytical solution and plot results
    fn = os.path.join(graphdir, 'VFI_analytical_mpc.pdf')
    plot_coefs_pfun(par, fn)

    # solve the household problem for given parameters
    vfun, pfun_sav = vfi(par)

    # recover consumption policy functions
    cah = (1.0 + par.r) * grid_a
    pfun_cons = cah - pfun_sav

    # Compute closed-form coefficients
    B = 1.0 / (1.0 - beta)
    A = beta/(1.0-beta)*np.log(beta) + np.log(1.0-beta) \
        + 1.0/(1.0-beta) * np.log(1.0 + r)
    A /= (1.0 - beta)
    vfun_analytical = A + B * np.log(grid_a)

    # Closed-form policy function coefficient
    kappa = (1.0 - par.beta)

    # Plot results
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(9, 3.5))

    xvalues = par.grid_a
    xticks = np.arange(0, a_max + 0.1, 1)

    axes[0].plot(xvalues, vfun, lw=2.0, alpha=0.7, label='Numerical')
    axes[0].plot(xvalues, vfun_analytical, lw=1.0, ls='-.', c='black',
                 label='Analytical')
    axes[0].legend(loc='lower right')
    ylim = axes[0].get_ylim()
    axes[0].set_ylim((max(vfun[0], -500.0), ylim[-1]))
    axes[0].set_xticks(xticks)
    axes[0].set_title('Value func. $V$')
    axes[0].set_xlabel('Assets')

    axes[1].plot(grid_a, pfun_sav, lw=2.0, alpha=0.7, label='Savings policy')
    axes[1].plot(grid_a, (1.0 - kappa)*cah, lw=1.0, ls='-.', c='black')
    axes[1].set_xticks(xticks)
    axes[1].set_title(r'Savings $a^{\prime}$')
    axes[1].set_xlabel('Assets')

    axes[2].plot(grid_a, pfun_cons, lw=2.0, alpha=0.7, label='Consumption policy')
    axes[2].plot(grid_a, kappa * cah, lw=1.0, ls='-.', c='black')
    axes[2].set_xticks(xticks)
    axes[2].set_title(r'Consumption $c$')
    axes[2].set_xlabel('Assets')

    fig.tight_layout()
    fn = os.path.join(graphdir, 'VFI_analytical_result.pdf')
    fig.savefig(fn)


def update_coefs(par, A, B):

    beta, r = par.beta, par.r

    A_upd = beta * A + beta * B * np.log(beta*B) \
            - (1.0 + beta*B) * np.log(1.0 + beta*B) \
            + (1.0 + beta*B) * np.log(1.0 + r)

    B_upd = 1.0 + beta * B

    return A_upd, B_upd


def f_objective(sav, cah, par, f_vfun):

    if sav < 0.0 or sav > cah:
        return np.inf

    # Consumption implied by savings level
    cons = cah - sav

    # Continuation value interpolated onto asset grid
    vcont = f_vfun(sav)

    # evaluate objective: log(c) + beta * V(a')
    obj = np.log(cons) + par.beta * vcont

    # We are running a minimiser, return negative of objective value
    return -obj


def vfi(par, tol=1e-5, maxiter=1000):

    N_a = len(par.grid_a)
    vfun = np.log(1.0 + par.r) + np.log(par.grid_a)
    vfun_upd = np.zeros(N_a)
    pfun_sav = np.zeros(N_a)

    # Coefficients A, B of closed-form value function in each iteration:
    # V_n(a) = A_n + B_n * log(a)
    Bn = 1.0
    An = np.log(1.0 + par.r)

    # Create figure, axes instance for plotting program
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    plot_iter = (0, ) + tuple(np.arange(4, 30, 5))
    # Plot only from this index onward
    ifrom = np.where(par.grid_a > 1.0e-2)[0][0]

    for it in range(maxiter):

        # Plot numerical solution for current iteration against
        # analytical solution
        if it in plot_iter:
            label = 'Numerical' if it == plot_iter[0] else None
            ax.plot(par.grid_a[ifrom:], vfun[ifrom:], color='steelblue',
                    lw=2.0, alpha=0.6, label=label)

            vfun_analytical = An + Bn * np.log(par.grid_a)
            label = 'Analytical' if it == plot_iter[0] else None
            ax.plot(par.grid_a[ifrom:], vfun_analytical[ifrom:], color='black',
                    lw=1.0, ls='-.', label=label)

            # Annotate point with current iteration number
            xy = par.grid_a[-1], vfun[-1]
            ax.annotate(f'{it+1}', xy, (5, 0), 'data', 'offset points',
                        va='center')

        # Update coefficients of analytical solution
        An, Bn = update_coefs(par, An, Bn)

        # create interpolating function for continuation value
        f_vfun = interp1d(par.grid_a, vfun, kind='linear', bounds_error=False,
                          fill_value='extrapolate', assume_sorted=True,
                          copy=False)

        for ia, a in enumerate(par.grid_a):
            # Solve maximization problem at given asset level
            # Cash-at-hand at current asset level
            cah = (1.0 + par.r) * a
            # Restrict maximisation to following interval:
            bounds = (1.0e-20, cah)
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
            msg = f'VFI: Converged after {it:3d} iterations, dV={diff:4.2e}'
            print(msg)
            break
        elif it == 1 or it % 10 == 0:
            msg = f'VFI: Iteration {it:3d}, dV={diff:4.2e}'
            print(msg)

    ax.set_xlim(par.grid_a[0], par.grid_a[-1] * 1.1)
    ylim = ax.get_ylim()
    ax.set_ylim((max(-40.0, ylim[0]), ylim[1]))
    ax.set_xlabel('Assets')
    ax.set_ylabel('Value function')
    ax.legend(loc='upper left', ncol=2)

    fn = os.path.join(graphdir, 'VFI_analytical_progress.pdf')
    fig.savefig(fn)

    return vfun, pfun_sav


def plot_coefs_vfun(par, filename, maxiter=1000, tol=1.0e-6):

    # initial values for n = 1
    A = np.log(1.0 + par.r)
    B = 1.0

    Aseq = [A]
    Bseq = [B]

    for i in range(maxiter):
        Anext, Bnext = update_coefs(par, A, B)

        diffA = abs(A - Anext)
        diffB = abs(B - Bnext)

        if diffA < tol and diffB < tol:
            break

        Aseq.append(Anext)
        Bseq.append(Bnext)

        A, B = Anext, Bnext

    Aseq = np.array(Aseq)
    Bseq = np.array(Bseq)

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(8, 3.5))

    iterations = np.arange(1, len(Aseq) + 1)

    axes[0].plot(iterations, Aseq, lw=2.0, label=r'$\chi_n$')
    axes[0].legend(loc='upper right')
    axes[0].set_xlabel('Log iteration')
    axes[0].set_xscale('log')

    axes[1].plot(iterations, Bseq, lw=2.0, label=r'$\varphi_n$')
    axes[1].legend(loc='lower right')
    axes[1].set_xlabel('Log iteration')
    axes[1].set_xscale('log')

    fig.tight_layout()
    fig.savefig(filename)


def plot_coefs_pfun(par, filename, maxiter=1000, tol=1.0e-6):

    # initial VFUN coefficients for n = 1
    A = np.log(1.0 + par.r)
    B = 1.0

    # initial values for n = 1
    mpc = 1.0 / (1.0 + par.beta) * (1.0 + par.r)

    mpc_seq = [mpc]

    for i in range(maxiter):
        Anext, Bnext = update_coefs(par, A, B)

        mpc_next = 1.0 / (1.0 + par.beta * Bnext) * (1.0 + par.r)

        diff = abs(mpc_seq[-1] - mpc_next)

        if diff < tol:
            break

        mpc_seq.append(mpc_next)

        A, B = Anext, Bnext

    mpc_seq = np.array(mpc_seq)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 3.5))

    iterations = np.arange(1, len(mpc_seq) + 1)

    ax.plot(iterations, mpc_seq, lw=2.0, label=r'$\kappa_n$')
    ax.legend(loc='upper right')
    ax.set_xlabel('Log iteration')
    ax.set_xscale('log')

    fig.tight_layout()
    fig.savefig(filename)


if __name__ == '__main__':
    main()
