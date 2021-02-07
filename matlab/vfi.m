
function [vfun, pfun_sav] = vfi(par, tol, maxiter)
% VFI   Solve HH problem using VFI with grid search.
%
%   [vfun, pfun_sav] = VFI(par,tol,maxiter) returns value function VFUN
%                   and policy function PFUN_SAV for next-period assets.
%   
%   Arguments:
%       par             structure containing parameters beta, r and w
%       tol             termination tolerance
%       maxiter         max. number of iterations
%
%   Returns:
%       vfun            value function for each point on the asset grid.
%       pfun_sav        policy function for next-period assets for each point 
%                       on the asset grid.
%
% Note: This implementation is slow, but should be more easy to
% understand.
%
% Author: Richard Foltyn

    % Set default values for optional arguments
    switch nargin
        case 1
            tol = 1.0e-6;
            maxiter = 1000;
        case 2
            maxiter = 1000;
    endswitch

  
    % start timer to calculate how long it takes to run VFI
    tstart = tic;

    N_a = par.N_a;
    % dimension of output arrays, which we define as column vectors
    dims = [N_a 1];

    % Initialize arrays where resulting value and policy functions will be stored.
    vfun = zeros(dims);
    % Array used to store updated value function
    vfun_upd = NaN(dims);
    % Array for policy function for next-period assets
    pfun_sav = NaN(dims);

    % Precompute cash-at-hand for each grid point
    % so we don't have to do that repeatedly in each loop iteration.
    cah = (1.0 + par.r) * par.grid_a + par.w;
  
    for iter = 1:maxiter
    
        % iterate through asset grid
        for ia = 1:N_a
            
            % current-period consumption
            cons = cah(ia) - par.grid_a;

            % current-period utility
            if par.gamma == 1.0
                u = log(cons);
            else
                u = (cons.^(1.0 - par.gamma) - 1.0) ./ (1.0 - par.gamma);
            endif
            
            % compute candidate values for all choices of a'
            vtry = u + par.beta * vfun;
            
            % logical index of feasible next-period states.
            feasible = par.grid_a < cah(ia);
            
            % set infeasible values to -Inf
            vtry(~feasible) = -Inf;

            % Find maximum value and the associated index on the asset grid
            [v_opt, imax] = max(vtry);

            % store optimal value as our policy choice for this iteration
            pfun_sav(ia) = imax;

            % Store optimal value in value function array
            vfun_upd(ia) = v_opt;
        end
    
        % check whether we have convergence, ie. difference to last iteration is
        % below desired tolerance level. If this is the case, exit the function,
        % otherwise proceed with next iteration.
        diff = max(max(abs(vfun-vfun_upd)));
        
        % update using newly computed value function
        vfun = vfun_upd;
          
        if diff < tol
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
