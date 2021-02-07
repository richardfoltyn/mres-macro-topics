function edist = markov_ergodic_dist(tm)
%   MARKOV_ERGODIC_DIST computes the ergodic distribution of a transition matrix.
%
%   EDIST = MARKOV_ERGODIC_DIST(TM) returns the ergodic distribution 
%       implied by the transition matrix TM, where any element TM(i,j)
%       represents the transition probability from state i to state j.

    N = size(tm, 1);
    
    mat = tm' - eye(N);
    mat(end,:) = 1.0;
    tmp = inv(mat);
    
    edist = tmp(:,end);
  
    % Perform some sanity checks
    assert(abs(sum(edist) - 1.0) < 1.0e-9)
    assert(all(edist >= 0.0))
    assert(all(edist <= 1.0))
    
    edist = edist / sum(edist);
  
end