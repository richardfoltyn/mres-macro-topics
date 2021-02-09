
function tightlayout(hfig,layout,add_margins,papersize)
%   TIGHTLAYOUT removes excessive margins around subplots.
%
%   TIGHTLAYOUT(HFIG) removes excessive margins for the given figure
%           (default: current figure).
%
%   TIGHTLAYOUT(HFIG,LAYOUT) removes excessive margins for the given 
%           figure, assuming that subplots are arranged using the
%           LAYOUT = [NROW, NCOL].
%
%   TIGHTLAYOUT(HFIG,LAYOUT,ADD_MARGINS) additionally increases the margins
%           of each subplot by the values given in the array 
%           ADD_MARGINS = [LEFT, BOTTOM, RIGHT, TOP]. Alternatively, ADD_MARGINS
%           can be a scaler if the same amount should be added to all sides.
%
%   TIGHTLAYOUT(HFIG,NROW,NCOL,ADD_MARGINS,PAPERSIZE) Additionally, sets 
%           the paper size to the given [width, height] values (only relevant 
%           when printing or saving).

    DEFAULT_ADD_MARGINS = repmat(0.1, [1 4]);

    switch nargin
        case 0
            hfig = gcf();
        case 1
            layout = [1 1];
            add_margins = DEFAULT_ADD_MARGINS;
        case 2
            add_margins = DEFAULT_ADD_MARGINS;
        case 3
            papersize = [];
        case 4
            % Admissible number of args, no action needed
        otherwise
            error('Unsupported number of arguments: %d', nargin);
    end
   
    
    % Replicate scalar margin across all four sides
    if isscalar(add_margins)
        add_margins = repmat(add_margins, [1 4]);
    end
    
    nrow = layout(1);
    ncol = layout(2);
    
    % Get children that are axes. In newer Matlab version, this will NOT
    % include legends.
    axes = findall(hfig, 'type', 'axes');

    naxes = length(axes);       % total number of axes in the figure
    
    % compute harmonized margins (left,bottom,right,top) across all
    % subplots. For this we find the max. for each margin.
    margins = zeros([1 4]);
    
    for k = 0:naxes-1
        hax = axes(naxes-k);
        
        % This does not work in Octave!
        if ~isOctave()
            set(hax, 'XLimSpec', 'Tight', 'YLimSpec', 'Tight');
        end
        inset = get(hax, 'TightInset');
        outer = get(hax, 'OuterPosition');

        inner = get(hax, 'Position');
        
        m = [inner(1) - outer(1) - inset(1), ...
             inner(2) - outer(2) - inset(2), ...
             outer(1) + outer(3) - inner(1) - inner(3) - inset(3), ...
             outer(2) + outer(4) - inner(2) - inner(4) - inset(4)];
             
        % inset values for bottom/top are clearly wrong since they are 
        % reported to be zero even in the presence of axes labels.
        % Replace zero values with the margin between outer and inner position.
        if inset(2) == 0.0
            inset(2) = m(2);
        end
        
        if inset(4) == 0.0
            inset(4) = m(4);
        end
         
        margins = max(margins, inset);
    end
    
    % short-hand to have less cluttered expressions
    am = add_margins;

    % Apply position to each subplot
    for k = 0:naxes-1
        i = floor(k/ncol) + 1;
        j = rem(k, ncol) + 1;
        
        hax = axes(naxes-k);
        
        left = (j - 1)/ncol + margins(1) + am(1);
        bottom = (nrow - i)/nrow + margins(2) + am(2);
        width = 1.0/ncol - (margins(1) + margins(3) + am(1) + am(3));
        height = 1.0/nrow - (margins(2) + margins(4) + am(2) + am(4));
        pos = [left bottom width height];
        set(hax, 'activepositionproperty', 'position');
        set(hax, 'Position', pos);
    end
    
    if size(papersize) > 0
        % set PaperSize if argument is present.
        set(hfig, 'PaperUnits', 'inches');
        set(hfig, 'PaperSize', papersize);
        set(hfig, 'PaperPositionMode', 'manual');
        pos = [0 0 papersize];
        set(hfig, 'PaperPosition', pos);
    end

end