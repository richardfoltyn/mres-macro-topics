function tightlayout(hfig,nrow,ncol)
    
    InSet = get(gca, 'Tightset');
    % In the following lines, constants 0.98, 0.97, 0.02, 0.75 can be modified according to your need.
    widthFig = (1-InSet(1)-InSet(3)) * 0.97; % Use 97% of the available width
    heightFig = (1-InSet(2)-InSet(4)) * 0.98; % Use 98% of the available height 
    StartX = InSet(1) + (1-InSet(1)-InSet(3)) * 0.02; % add left margin, 2% of the width
    StartY = InSet(2) + (1-InSet(2)-InSet(4)) * 0.02; % add bottom margin
    naxes = length(hfig.Children); % total number of axes in the figure
    
    for k = 1:naxes
        
        i = floor(k/ncol) + 1;
        j = rem(k, ncol);
        
        if j==0
            j=ncol; 
            i = floor(k/ncol); 
        end
        
        hax = hfig.Children(naxes + 1 - k); 
        % For each subplot, set the position:
        set(hax, 'Position', [StartX + widthFig * (j-1)/ncol, ...
                              StartY + heightFig * (nrow - i)/nrow,...
                              (widthFig/ncol)*0.75, ...
                              (heightFig/nrow)*0.75]);
    end

end