function showModelX(M, yrange)
%%--------------------------------------------------
%% Show the dx over a set of constant y values.
%% yrange is a list of indices into the unique y 
%% values in the warp grid w.
%%
%% 2020-05-19   Alex Szalay
%%--------------------------------------------------
    %
    uy = unique(M.y);
    ny = numel(uy);
    %
    if (min(yrange)>ny | max(yrange)<1)
        fprintf('yrange is outside (1:%d)\n', ny);
        return
    end
    %
    ylabel('dx')
    xlabel('x')
    xlim([0 19999]);
    grid on
    box on
    %
    if (numel(yrange)==1)
        s = sprintf('y=%d',uy(yrange));
    else
        s = sprintf('y=(%d..%d)',min(uy(yrange)),max(uy(yrange)));
    end
    title(s);
    %
    hold on
    %
    for i=yrange
        if (i<1 | i>ny)
            continue
        end
        y = uy(i);
        ww = M(M.y==y,:);
        %
        plot(ww.x,ww.dx,'-b');
        plot(ww.x,ww.mdx,'-r');
    end
    hold off
    shg
    %
end
