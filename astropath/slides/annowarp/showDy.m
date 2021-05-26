function showDy(w, xrange)
%%--------------------------------------------------
%% Show the dx and its model over a set of constant y values.
%% yrange is a list of indices into the unique y 
%% values in the warp grid w.
%%
%% 2020-05-19   Alex Szalay
    %
    figure(5);
    subplot(3,1,1);
    showDyRaw(w,xrange,0);
    %
    subplot(3,1,2);
    showDyRaw(w,xrange,1);
    %
    subplot(3,1,3);
    showDyRaw(w,xrange,2);
end


function showDyRaw(w, xrange,flag)
%%--------------------------------------------------
%% Show the dx over a set of constant y values.
%% yrange is a list of indices into the unique y 
%% values in the warp grid w.
%%
%% 2020-05-19   Alex Szalay
%%--------------------------------------------------
    %
    uy = unique(w.y);
    ux = unique(w.x);
    nx = numel(ux);
    ny = numel(uy);
    %
    if (min(xrange)>nx | max(xrange)<1)
        fprintf('yrange is outside (1:%d)\n', nx);
        return
    end
    %
    ylabel('dy')
    xlabel('y')
    xlim([uy(1) uy(end)]);
    %ylim([-20 10]);
    grid on
    box on
    %
    if (numel(xrange)==1)
        s = sprintf('x=%d',ux(xrange));
    else
        s = sprintf('y=(%d..%d)',min(ux(xrange)),max(ux(xrange)));
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
        ww = w(w.x==x,:);
        %
        if (numel(ww.y)<5)
            continue
        end
        %{
        gps = unique(ww.gc);       
        for g=1:numel(gps)
            wg = ww(ww.gc==gps(g),:);
            if (numel(wg.n)>5)
                plot(wg.x,wg.dx,'-');
            end
        end
        %}
        showgroups(ww,flag);
    end
    hold off
    shg
    %
end


function showgroups(ww,flag)
    gps = unique(ww.gc);       
    for g=1:numel(gps)
        wg = ww(ww.gc==gps(g),:);
        Y = wg.y;
        if (flag==0)
            DY = wg.dy;
        elseif(flag==1)
            DY = wg.mdy;
        elseif (flag==2)
            DY = wg.ry;
        end
        if (numel(wg.n)>5)
            plot(Y,DY,'-');
        end
    end
end
        


function showDyModel(w,xcut,a,e,g)
%%-------------------------------------------------------
%% Overlay model with dx cut at a given y
%% w: the table with the displacement fields
%% ycut: the value of y at the cut, has to be from w.y
%% a: the falling slope of the ramp
%% e: the incremental step at the stitching edges
%% g: the global ofset of the dx value at the origin
%% Example:
%%   showDxModel(w,8400,-0.5,7,5.6);
%%
%% 2020-05-20   Alex Szalay
%%-------------------------------------------------------
    %
    % extract data along the cut
    %
    ww = w(w.x==xcut,:);
    %
    hold on
    xlabel('y');
    ylabel('dy');
    title('x=8400');
    grid on
    box on
    plot(ww.x,ww.dx,'.-');
    %
    GX = 1400;
    n = 0;
    h = e;
    for i=1:180
        x(i) = 100*i;
        if (mod(i,floor(GX/100))==0)
            h = h+g;
        end
        h = h + a;
        y(i) = h;
    end
    plot(x,y,'-');
    %
    hold off
    shg
    %
end

