function grid = powerspace(xmin, xmax, n, xp)
    %   POWERSPACE  Creates a non-uniformly spaced grid.
    %
    %   GRID = POWERSPACE(XMIN, XMAX, N, XP) returns a grid if N points 
    %       on the interval [XMIN,XMAX]. 
    %       The argument XP controls the point density, with XP > 1 allocating 
    %       more points at the lower and of the grid, and XP < 1 allocating more 
    %       points at the upper end.
    
    u = linspace(0.0, 1.0, n);
    
    grid = xmin + (xmax - xmin) .* u.^xp;
    % Replace last element to prevent rounding errors
    grid(end) =  xmax;
end
