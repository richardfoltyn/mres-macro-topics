
function [vfun, pfun_sav] = vfi_risk_interp(par, tol, maxiter, method)
% VFI   Solve HH problem with labour income risk using VFI with grid search.
%
%   [VFUN, PFUN_SAV] = VFI_RISK_INTERP(PAR) returns the value function and savings
%       policy function for the problem parametrised by PAR.
%
%   [VFUN, PFUN_SAV] = VFI_RISK_INTERP(PAR,TOL,MAXITER) returns the value function and 
%       savings policy function for the problem parametrised by PAR,
%       using the termination tolerance TOL and a max. number of iterations
%       given by MAXITER.
%
%   [VFUN, PFUN_SAV] = VFI_RISK_INTERP(PAR,TOL,MAXITER,METHOD) additionally
%       uses the given interpolation METHOD (linear, cubic, etc.). 
%
% Note: This implementation is slow, but should be more easy to
% understand.
%
% Author: Richard Foltyn

    % Set default values for optional arguments
    switch nargin
        case 1
            % Only PAR passed, assume default TOL and MAXITER
            tol = 1.0e-6;
            maxiter = 1000;
            % default interpolation method
            method = 'linear';
        case 2
            % Two arguments passed, assume default MAXITER
            maxiter = 1000;
            % default interpolation method
            method = 'linear';
        case 3
            % default interpolation method
            method = 'linear';
        case 4
            % All arguments present
        otherwise
            error('Invalid number of arguments: %d', nargin);
    end

  
    % start timer to calculate how long it takes to run VFI
    tstart = tic;

    N_a = par.N_a;
    N_y = par.N_y;
    % dimension of output arrays
    dims = [N_a N_y];

    % Initialize arrays where resulting value and policy functions will 
    % be stored.
    % Current guess for the value function
    vfun = zeros(dims);
    % Updated guess for the value function
    vfun_upd = NaN(dims);
    % Array for policy function for next-period assets
    pfun_sav = NaN(dims);
  
    for iter = 1:maxiter
    
        % Iterate over labour states
        for iy = 1:N_y
            
            % Expected continuation value E[V(a',y')|y] for current y
            vcont = vfun * par.tm_y(iy,:)';
            
            % iterate through asset grid
            for ia = 1:N_a
                % Assets at current grid point
                assets = par.grid_a(ia);

                % Cash-at-hand at current grid point
                cah = (1.0 + par.r) * assets + par.grid_y(iy);

                % Restrict maximisation to feasible interval
                [sav, fval] = fminbnd(@(x) objective(x, cah, par, vcont, method), 0.0, cah);

                % value function is negative of minimizer objective
                vfun_upd(ia,iy) = -fval;

                % Optimal savings choice
                pfun_sav(ia,iy) = sav;
            end
        end
        
        % check whether we have convergence, ie. difference to last iteration is
        % below desired tolerance level. If this is the case, exit the function,
        % otherwise proceed with next iteration.
        diff = max(max(abs(vfun-vfun_upd)));
        
        % update using newly computed value function
        vfun = vfun_upd;
          
        if diff < tol
            % Desired tolerance level achieved, terminate VFI.
            tend = toc(tstart);
            fprintf("Converged after %d iterations in %4.1f sec.; dV=%.2e\n", ...
                iter, tend, diff);
            return;
        else
            % print progress in the first iteration, and thereafter every 10 
            % iterations.
            if (mod(iter, 10) == 0) || (iter == 1)
                fprintf("VFI: iteration %3d; dV = %.2e\n", iter, diff);
            end
        end
    end
  
    warning("Exceeded max number of iterations; dV=%.2e\n", diff);

end


function fval = objective(sav, cah, par, vcont, method)
% OBJECTIVE evaluates the household's objective at a given savings level.
%
%   FVAL = OBJECTIVE(SAV,CAH,PAR,VCONT,METHOD) returns the NEGATIVE expected
%       utility for savings choice SAV, cash-at-hand CAH, model parameters
%       PAR and the continuation value VCONT which is defined on the
%       asset grid. The continuation value is interpolated using the given
%       METHOD (linear, cubic, etc.).

    % Exit immediately if the savings choice is infeasible
    if sav < 0.0 || sav >= cah
        fval = Inf;
        return
    end
    
    % Consumption implied by cash-at-hand and savings level
    cons = cah - sav;
    
    % Interpolated continuation value
    vnext = interp1(par.grid_a, vcont, sav, method, 'extrap');
    
    % current-period utility
    if par.gamma == 1.0
        % Log preferences with RRA = 1
        u = log(cons);
    else
        % General CRRA
        u = (cons^(1.0 - par.gamma) - 1.0) / (1.0 - par.gamma);
    end

    % Objective evaluated at current savings level
    v = u + par.beta * vnext;
    
    % Minimizer: need to return negative expected utility
    fval = -v;
end
