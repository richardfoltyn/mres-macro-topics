
function [pfun_cons, pfun_sav] = egm_IH(par, tol, maxiter)
% EGM_IH   Solve the infinite-horizon HH problem with constant labour using EGM.
%
%   [PFUN_CONS, PFUN_SAV] = EGM_IH(PAR,TOL,MAXITER) returns the consumption and 
%       savings policy functions for the problem parametrised by PAR,
%       using the termination tolerance TOL and a max. number of iterations
%       given by MAXITER.
%
%   Note: This implementation uses the same grid for exogenous savings
%       and beginning-of-period assets. This is not required, but avoids
%       unnecessary interpolation steps.
%
% Author: Richard Foltyn
  
    % start timer to calculate how long it takes to run EGM
    tstart = tic;
   
    N_a = par.N_a;
    % Dimensions of arrays created below
    dims = [N_a 1];

    % Precompute cash-at-hand for each asset/savings grid point
    % so we don't have to do that repeatedly in each loop iteration.
    cah = (1.0 + par.r) * par.grid_a + par.y;
    
    % Initialise arrays to hold optional consumption as a function of 
    % savings brought over from previous period.
    
    % Initial guess for consumption policy: consume entire cash-at-hand
    pfun_cons = cah;
    % Array to store updated consumption policy function
    pfun_cons_upd = NaN(dims);
    % Array to store savings policy function
    pfun_sav = NaN(dims);
  
    for iter = 1:maxiter
               
        % Marginal utility tomorrow as a function of
        % savings today (which are tomorrow's assets).
        %   mu = u'(C(a'))
        mu = pfun_cons.^(-par.gamma);

        % Compute right-hand side of Euler equation
        ee_rhs = par.beta * (1.0 + par.r) * mu;

        % Invert Euler equation to get optimal consumption today as
        % a function of today's savings.
        cons_sav = ee_rhs.^(-1.0/par.gamma);

        % Use budget constraint to get beginning-of-period assets
        % as a function of today's savings.
        assets_sav = (cons_sav + par.grid_a - par.y) / (1.0 + par.r);

        % The arrays (ASSETS_SAV, CONS_SAV) represent the updated
        % consumption policy function, defined on the endogenous asset grid.
        % Interpolate results back onto exogenous asset/savings grid.
        pfun_cons_upd(:) = interp1(assets_sav, cons_sav, ...
            par.grid_a, 'linear', 'extrap');

        % Fix consumption policy for states in which HH does not
        % save anything. Since HH does not save anything, it must
        % consume its entire cash-at-hand in those states.
        % ASSETS_SAV(1) is the largest asset level for which household
        % does not save anything.
        ia = find(par.grid_a < assets_sav(1));
        if size(ia) > 0
            pfun_cons_upd(ia) = cah(ia);
        end
        
        % Make sure that consumption policy satisfies constraints.
        % This will terminate the program whenever consumption is either
        % negative or exceeds available resources.
        assert(all(pfun_cons_upd >= 0.0) && all(pfun_cons_upd <= cah));
        
        % check whether we have convergence, ie. difference to last iteration is
        % below desired tolerance level. If this is the case, exit the function,
        % otherwise proceed with next iteration.
        diff = max(abs(pfun_cons - pfun_cons_upd));
        
        % Copy over updated consumption policy for next iteration
        pfun_cons(:) = pfun_cons_upd;

        % Update current guess for savings policy function defined on
        % beginning-of-period assets grid.
        pfun_sav(:) = cah - pfun_cons;
          
        if diff < tol
            % Desired tolerance level achieved, terminate iteration.
            tend = toc(tstart);
            fprintf("EGM: Converged after %d iterations in %.2f sec.; dC=%.2e\n", ...
                iter, tend, diff);
            return;
        else
            % print progress in the first iteration, and thereafter every 10 
            % iterations.
            if (mod(iter, 10) == 0) || (iter == 1)
                fprintf("EGM: iteration %3d; dC = %.2e\n", iter, diff);
            end
        end
    end
  
    warning("EGM: Exceeded max number of iterations; dC=%.2e\n", diff);

end
