function showQuiver(W)
%%-------------------------------------------------
%% show the quiver plot of the displacement field.
%% Use different color for each region.
%%
%% 2020-05-18   Alex Szalay
%%-------------------------------------------------
    %
    if (ishandle(3))
        close(3);
    end
    %
    figure(3);
    hold on
    %
    gps = unique(W.gc);
    clr = colormap(lines);
    %
    for i=1:numel(gps)
        w = W(W.gc==gps(i),:);        
        quiver(w.x,w.y,w.dx,w.dy,'Color',clr(mod(i,255)+1,:));
    end
    %
    %hold on
    box on
    shg
    %
end