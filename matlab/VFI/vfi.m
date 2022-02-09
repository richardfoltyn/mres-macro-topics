
function [vfun, pfun_ia] = vfi(par, tol, maxiter)
% VFI   Solve HH problem using VFI with grid search.
%
%   [VFUN, PFUN_IA] = VFI(PAR,TOL,MAXITER) returns the value function and 
%       savings policy function for the problem parametrised by PAR,
%       using the termination tolerance TOL and a max. number of iterations
%       given by MAXITER.
%
% Return values:
%   vfun        Array containing the value function defined on the
%               asset grid.
%   pfun_ia     Array containing the indices of optimal assets next
%               period.
%
% Note: This implementation is slow, but should be more easy to
% understand.
%
% Author: Richard Foltyn
  
    % start timer to calculate how long it takes to run VFI
    tstart = tic;

    N_a = par.N_a;
    % dimension of output arrays, which we define as column vectors
    dims = [N_a 1];

    % Initialize arrays where resulting value and policy functions will 
    % be stored.
    % Current guess for the value function
    vfun = zeros(dims);
    % Updated guess for the value function
    vfun_upd = NaN(dims);
    % Array for index for next-period assets;
    % The type uint32 declares this to be 32bit integer array.
    pfun_ia = zeros(dims, 'uint32');

    % Precompute cash-at-hand for each grid point
    % so we don't have to do that repeatedly in each loop iteration.
    cah = (1.0 + par.r) * par.grid_a + par.y;
  
    for iter = 1:maxiter
    
        % iterate through asset grid
        for ia = 1:N_a
            
            % current-period consumption for all possible savings choices,
            % irrespective of whether they are feasible or not.
            cons = cah(ia) - par.grid_a;

            % current-period utility
            if par.gamma == 1.0
                % Log preferences with RRA = 1
                u = log(cons);
            else
                % General CRRA
                u = (cons.^(1.0 - par.gamma) - 1.0) / (1.0 - par.gamma);
            end
            
            % compute candidate values for all choices of a':
            % V(a) = u(c) + beta * V(a')
            vtry = u + par.beta * vfun;
            
            % Logical array identifying feasible choices that satisfy the
            % budget constraint.
            feasible = par.grid_a < cah(ia);
            
            % set infeasible values to -Inf
            vtry(~feasible) = -Inf;

            % Find maximum value and the associated index on the asset grid
            [v_opt, imax] = max(vtry);

            % store optimal value as our policy choice for this iteration
            pfun_ia(ia) = imax;

            % Store optimal value in value function array
            vfun_upd(ia) = v_opt;
        end
    
        % check whether we have convergence, ie. difference to last iteration is
        % below desired tolerance level. If this is the case, exit the function,
        % otherwise proceed with next iteration.
        diff = max(abs(vfun-vfun_upd));
        
        % update using newly computed value function
        vfun = vfun_upd;
          
        if diff < tol
            % Desired tolerance level achieved, terminate VFI.
            tend = toc(tstart);
            fprintf("VFI: Converged after %d iterations in %.2f sec.; dV=%.2e\n", ...
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
  
    warning("VFI: Exceeded max number of iterations; dV=%.2e\n", diff);

end
