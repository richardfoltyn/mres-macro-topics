
function [vfun, pfun_sav] = vfi(par, tol, maxiter)
% Template function to solve household problems, e.g. using VFI or EGM.
%
% Return values:
%   vfun        Array containing the value function defined on the
%               asset grid.
%   pfun_sav    Array containing the savings policy function (optimal
%               next-period asset choices)
%
% Arguments:
%   par         Struct containing model parameters and grids
%   tol         Termination tolerance
%   maxiter     Max. number of iterations.
%
% Author: Richard Foltyn
  
    % start timer to calculate how long it takes to solve HH problem.
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
    % Array to store next-period assets;
    pfun_sav = zeros(dims);

  
    for iter = 1:maxiter
    
        % iterate through asset grid
        for ia = 1:N_a
            
        end
    
        % check whether we have convergence, ie. difference to last 
        % iteration is below desired tolerance level. 
        % If this is the case, exit the function, otherwise proceed 
        % with next iteration.
        diff = max(abs(vfun-vfun_upd), [], 'all');
        
        % update using newly computed value function
        vfun = vfun_upd;
          
        if diff < tol
            % Desired tolerance level achieved, terminate.
            tend = toc(tstart);
            fprintf("VFI: Converged after %d iterations in %.2f sec.; dV=%.2e\n", ...
                iter, tend, diff);
            return;
        else
            % print progress in the first iteration, and thereafter every 
            % 10 iterations.
            if (mod(iter, 10) == 0) || (iter == 1)
                fprintf("VFI: iteration %3d; dV = %.2e\n", iter, diff);
            end
        end
    end
  
    warning("VFI: Exceeded max number of iterations; dV=%.2e\n", diff);

end
