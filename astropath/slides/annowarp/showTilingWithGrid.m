function w = showTilingWithGrid(W,E,flag)
%%-------------------------------------------------
%% Show the contour map of the warp fields
%% with the tiling grid outlines overlaid.
%% Everything is in ipix coordinates
%% flag controls which maps get plotted
%%  0: raw data (dx,dy)
%%  1: model    (mdx, mdy)
%%  2: res      (rx,ry)
%%
%% 2020-05-18   Alex Szalay
%%-------------------------------------------------
    %
    figure(7);
    %
    sz=100;
    %
    gps = unique(W.gc);
    %
    n=1;
    m=2;
    k=1;
    subplot(n,m,k);
        hold on
        for i=1:numel(gps)
            w = W(W.gc==gps(i),:);
            [DX,DY] = selectDelta(w,flag);
            if (numel(w.n)>10)
                surfir(double(w.x)+sz/2,double(w.y)+sz/2,DX);
            end
        end    
        hold off
        %
        title('dx')
        xlabel('x')
        ylabel('y')
        view(360,-90);
        axis equal
        axis([E.ex(1) E.ex(end) E.ey(1) E.ey(end)]);        
        box on
        grid off
        k = k+1;
        hold on
        showgrid(E);
        colorbar
        hold off
    subplot(n,m,k);
        hold on
        for i=1:numel(gps)
            w = W(W.gc==gps(i),:);
            [DX,DY] = selectDelta(w,flag);
            if (numel(w.n)>10)
                surfir(double(w.x)+sz/2,double(w.y)+sz/2,DY);
            end
        end    
        hold off
        %
        title('dy')
        xlabel('x')
        ylabel('y')
        view(360,-90);
        axis equal
        axis([E.ex(1) E.ex(end) E.ey(1) E.ey(end)]);        
        box on 
        grid off
        k = k+1;
        hold on
        showgrid(E);
        colorbar
        hold off        
    shg
    %
end

function [DX,DY] = selectDelta(W,flag)
    if (flag==0)
        DX = W.dx;
        DY = W.dy;
    end
    if (flag==1)
        DX = W.mdx;
        DY = W.mdy;
    end
    if (flag==2)
        DX = W.rx;
        DY = W.ry;
    end
    %
end
    

function showgrid(E)
    %
    clr = [0.25,0.25,0.25];
    z = -25000;
    %
    ex = 0.999*E.ex;
    for i=1:numel(ex)
        plot3([ex(i),ex(i)],[E.ey(1),E.ey(end)],[z,z],':','Color',clr);
    end
    %
    ey = 0.999*E.ey;
    for i=1:numel(ey)
        plot3([E.ex(1),E.ex(end)],[ey(i),ey(i)],[z,z],':','Color',clr);
    end
end