
function [pfun_cons, pfun_sav] = egm_IH(par, tol, maxiter)
% EGM_IH   Solve the infinite-horizon HH problem using EGM.
%
%   [PFUN_CONS, PFUN_SAV] = EGM_IH(PAR) returns the consumption and savings
%       policy functions for the problem parametrised by PAR.
%
%   [PFUN_CONS, PFUN_SAV] = EGM_IH(PAR,TOL,MAXITER) returns the consumption and 
%       savings policy functions for the problem parametrised by PAR,
%       using the termination tolerance TOL and a max. number of iterations
%       given by MAXITER.
%
%   Note: This implementation uses the same grid for exogenous savings
%       and beginning-of-period assets. This is not required, but spares
%       is unnecessary interpolation steps.
%
% Author: Richard Foltyn

    % Set default values for optional arguments
    switch nargin
        case 1
            % Only PAR passed, assume default TOL and MAXITER
            tol = 1.0e-6;
            maxiter = 1000;
        case 2
            % Two arguments passed, assume default MAXITER
            maxiter = 1000;
        case 3
            % All arguments present
        otherwise
            error('Invalid number of arguments: %d', nargin);
    end

  
    % start timer to calculate how long it takes to run VFI
    tstart = tic;
    
    % Check whether we have a problem with income risk. In this case we 
    % assume that the field GRID_Y and TM_Y are fields in the structure
    % PAR and store the states and transition matrix of income shocks.
    % If these fields are not present, we assume that income is
    % deterministic and equal to 1.0 each period.
    
    if isfield(par, 'grid_y') && isfield(par, 'tm_y')
        grid_y = par.grid_y;
        tm_y = par.tm_y;
        N_y = length(par.grid_y);
    else
        grid_y = ones(1,1);     % degenerate income state vector with 1 element
        tm_y = ones(1,1);       % degenerate 1-by-1 transition matrix
        N_y = 1;
    end

    N_a = length(par.grid_a);

    % Precompute cash-at-hand for each asset/savings grid point
    % so we don't have to do that repeatedly in each loop iteration.
    cah = (1.0 + par.r) * par.grid_a + grid_y';
    
    % Initialize arrays to hold optional consumption as a function of 
    % savings brought over from previous period.
    
    % Initial guess for consumption policy: consume entire cash-at-hand
    pfun_cons = cah;
    pfun_cons_upd = NaN([N_a, N_y]);
  
    for iter = 1:maxiter
    
        % Iterate over income levels
        for iy = 1:N_y
               
            % Expected marginal utility tomorrow as a function of
            % savings today (which are tomorrow's assets).
            mu = pfun_cons.^(-par.gamma) * tm_y(iy,:)';

            % Compute right-hand side of Euler equation
            ee_rhs = par.beta * (1.0 + par.r) * mu;

            % Invert Euler equation to get optimal consumption today as
            % a function of today's savings.
            cons_sav = ee_rhs.^(-1.0/par.gamma);

            % Use budget constraint to get beginning-of-period assets
            % as a function of today's savings.
            assets_sav = (cons_sav + par.grid_a - grid_y(iy)) / (1.0 + par.r);

            % Interpolate results back onto exogeneous asset/savings grid.
            pfun_cons_upd(:,iy) = interp1(assets_sav, cons_sav, ...
                par.grid_a, 'linear', 'extrap');

            % Fix consumption policy for states in which HH does not
            % save anything.
            ia = find(par.grid_a < assets_sav(1));
            if size(ia) > 0
                pfun_cons_upd(ia,iy) = cah(ia,iy);
            end
        end
        
        % Make sure that consumption policy satisfies constraints
        assert(all(all(pfun_cons_upd >= 0.0)) && all(all(pfun_cons_upd <= cah)));
        
        % check whether we have convergence, ie. difference to last iteration is
        % below desired tolerance level. If this is the case, exit the function,
        % otherwise proceed with next iteration.
        diff = max(max(abs(pfun_cons - pfun_cons_upd)));
        
        % update using newly computed value function
        pfun_cons = pfun_cons_upd;

        % Update current guess for savings policy function defined on
        % beginning-of-period assets grid.
        pfun_sav = cah - pfun_cons;
          
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
