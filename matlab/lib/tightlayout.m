
function tightlayout(hfig,layout,papersize)
%   TIGHTLAYOUT removes excessive margins around subplots.
%
%   TIGHTLAYOUT(HFIG) removes excessive margins for the given figure
%           (default: current figure).
%
%   TIGHTLAYOUT(HFIG,LAYOUT) removes excessive margins for the given 
%           figure, assuming that subplots are arranged using the
%           LAYOUT = [NROW, NCOL].
%
%   TIGHTLAYOUT(HFIG,NROW,NCOL,PAPERSIZE) removes excessive margins
%           for the given figure, assuming that subplots are arranged 
%           using the LAYOUT = [NROW, NCOL]. Additionally, the paper size
%           is set to the given [width, height] values (only relevant when
%           printing or saving).

    switch nargin
        case 0
            hfig = gcf();
        case 1
            layout = [1 1];
        case 2
            % Admissible number of args, no action needed
        case 3
            % Admissible number of args, no action needed
        otherwise
            error('Unsupported number of arguments: %d', nargin);
    end
    
    nrow = layout(1);
    ncol = layout(2);
    
    % Get children that are axes. In newer Matlab version, this will NOT
    % include legends.
    axes = findall(hfig, 'type', 'axes');

    naxes = length(axes);       % total number of axes in the figure
    
    % compute harmonized margins (left,bottom,top,right) across all
    % subplots. For this we find the max. for each margin.
    dleft = 0.0;
    dbottom = 0.0;
    dright = 0.0;
    dtop = 0.0;
    
    % Positioning of axes within figure seems to work somewhat differently
    % in Octave, skip as it does not yield the desired results.
    if ~isOctave()
        for k = 1:naxes
            hax = axes(k);
            % This does not work in Octave!
            set(hax, 'XLimSpec', 'Tight', 'YLimSpec', 'Tight');
            
            outer = get(hax, 'OuterPosition');
            inner = get(hax, 'Position');
            diff = inner - outer;
            
            dleft = max(dleft, diff(1));
            dbottom = max(dbottom, diff(2));
            % Max. right margin: 
            % (outer(1) + outer(3)) - (inner(1) + inner(3))
            %   = (outer(1) - inner(1)) + (outer(3) - inner(3))
            dright = max(dright, -diff(1) - diff(3));
            % Max. top margin: 
            % (1 - outer(2) - outer(4)) - (1 - inner(2) - inner(4)) 
            %   = (inner(2) - outer(2)) + (inner(4) - outer(4))
            dtop = max(dtop, diff(2) + diff(4));
        end
        
        % Apply position to each subplot
        for k = 0:naxes-1
            i = floor(k/ncol) + 1;
            j = rem(k, ncol) + 1;
            
            hax = axes(naxes-k);
            
            left = (j - 1)/ncol + dleft;
            bottom = (nrow - i)/nrow + dbottom;
            width = 1.0/ncol - dleft - dright;
            height = 1.0/nrow - dtop - dbottom;
            pos = [left bottom width height];
            set(hax, 'activepositionproperty', 'position');
            set(hax, 'Position', pos);
        end
    end
    
    if nargin == 3
        % set PaperSize if argument is present.
        set(hfig, 'PaperUnits', 'inches');
        set(hfig, 'PaperSize', papersize);
        set(hfig, 'PaperPositionMode', 'manual');
        pos = [0 0 papersize];
        set(hfig, 'PaperPosition', pos);
    end

end