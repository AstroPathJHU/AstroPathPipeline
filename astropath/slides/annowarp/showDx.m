function showDx(w, yrange)
%%--------------------------------------------------
%% Show the dx and its model over a set of constant y values.
%% yrange is a list of indices into the unique y 
%% values in the warp grid w.
%%
%% 2020-05-19   Alex Szalay
    %
    figure(5);
    subplot(2,1,1);
        showDxRaw(w,yrange);
%     subplot(2,1,2);
%         showDxModel(w,8400,-0.5,7,5.6);
    %
end


function showDxRaw(w, yrange)
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
    ny = numel(uy);
    %
    if (min(yrange)>ny | max(yrange)<1)
        fprintf('yrange is outside (1:%d)\n', ny);
        return
    end
    %
    ylabel('dx')
    xlabel('x')
    xlim([ux(1) ux(end)]);
    ylim([-20 10]);
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
        ww = w(w.y==y,:);
        %
        if (numel(ww.x)<5)
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
        showgroups(ww);
    end
    hold off
    shg
    %
end


function showgroups(ww)
    gps = unique(ww.gc);       
    for g=1:numel(gps)
        wg = ww(ww.gc==gps(g),:);
        if (numel(wg.n)>5)
            plot(wg.x,wg.dx,'-');
        end
    end
end
        


function showDxModel(w,ycut,a,e,g)
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
    ww = w(w.y==ycut,:);
    %
    hold on
    xlabel('x');
    ylabel('dx');
    title('y=8400');
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

