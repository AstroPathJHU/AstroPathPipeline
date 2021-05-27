function showModelY(M, xrange)
%%--------------------------------------------------
%% Show the dx over a set of constant y values.
%% yrange is a list of indices into the unique y 
%% values in the warp grid w.
%%
%% 2020-05-19   Alex Szalay
%%--------------------------------------------------
    %
    ux = unique(M.x);
    nx = numel(ux);
    %
    if (min(xrange)>nx | max(xrange)<1)
        fprintf('yrange is outside (1:%d)\n', nx);
        return
    end
    %
    ylabel('dy')
    xlabel('y')
    xlim([0 19999]);
    grid on
    box on
    %
    if (numel(xrange)==1)
        s = sprintf('x=%d',ux(xrange));
    else
        s = sprintf('x=(%d..%d)',min(ux(xrange)),max(ux(xrange)));
    end
    title(s);
    %
    hold on
    %
    for i=xrange
        if (i<1 | i>nx)
            continue
        end
        x = ux(i);
        ww = M(M.x==x,:);
        plot(ww.y,ww.dy,'-b');
        plot(ww.y,ww.mdy,'-r');
    end
    hold off
    shg
    %
end

