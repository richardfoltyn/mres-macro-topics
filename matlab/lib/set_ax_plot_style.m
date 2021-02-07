function set_ax_plot_style(ax, aspect)
%   SET_AX_PLOT_STYLE applies common style settings to axes object.
%
%   SET_AX_PLOT_STYLE(AX) applies settings to specific axes (default:
%       current axes)
%   SET_AX_PLOT_STYLE(AX,ASPECT) sets the given aspect ratio.

    if nargin == 0
        ax = gca();
        aspect = 1.0;
    elseif nargin == 1
        aspect = 1.0;
    end
    
    set(ax, 'FontWeight', 'normal', 'FontSize', 9, 'FontName', 'Times');
    
    grid(ax, 'on');
    set(ax, 'GridLineStyle', ':');
    set(ax, 'GridAlpha', 0.4);
    
    pbaspect(ax, [aspect 1.0 1.0]);

end
