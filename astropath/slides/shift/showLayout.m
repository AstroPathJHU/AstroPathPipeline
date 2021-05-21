function C = showLayout(C,flag)
%%-------------------------------------------------
%% show the layout of the high-powered fields
%%  flag=0 just shows the image
%%  flag=1 shows the HPF outlines
%%  flag=2 shows the scanning pattern
%%  flag=3 shows the primary areas as well
%%
%% Alex Szalay, Baltimore, 2018-11-14
%%-------------------------------------------------
    %
    logMsg(C,'showLayout');
    %
    imshow(C.qimg);
    set (gca,'Ydir','normal');
    hold on
    %
    % field offsets and sizes all in microns
    %
    pw = C.fwidth/C.pscale;  % in microns
    ph = C.fheight/C.pscale;  % in microns
    %
    if (flag==0)
        return
    end
    %
    if (flag==1 | flag==2)
        fprintf('Flag==%d\n',flag);
        tx = C.qpscale*(C.R.x-C.xposition);
        ty = C.qpscale*(C.R.y-C.yposition);
        %
        tw = C.qpscale*pw;
        th = C.qpscale*ph;
        %
        % show the field outlines
        %
        N = numel(tx);
        if (flag>0)
            for i=1:N
                rect2lines(tx(i),ty(i),tx(i)+tw,ty(i)+th,'r-');
            end
        end
    end
    %{
    if (flag==3)
        fprintf('Flag==%d\n',flag);
        tx = C.qpscale*(C.H.x)/C.pscale;
        ty = C.qpscale*(C.H.y)/C.pscale;
        %
        tw = C.qpscale*pw/C.pscale;
        th = C.qpscale*ph/C.pscale;
        %
        % show the field outlines
        %
        N = numel(tx);
        if (flag>0)
            for i=1:N
                rect2lines(tx(i),ty(i),tx(i)+tw,ty(i)+th,'r-');
            end
        end        
    end
    %}
    %
    % show the scanning  pattern
    %
    m = 6;
    if (flag==2)        
        %
        plot(tx+tw/2,ty+th/2,'gs:','MarkerSize',3);
        plot(tx(1)+tw/2,ty(1)+th/2,'r>',...
            'MarkerFaceColor','r','MarkerSize',m);
        plot(tx(N)+tw/2,ty(N)+th/2,'g^',...
            'MarkerFaceColor','m','MarkerSize',m);
    end
    %
    % show the primary regions in green
    %    
    if (flag==3)
        %
        fprintf('flag==%d\n',flag);
        if (isfield(C,'H')==0)
            logMsg(C,'C.H is not present');
            return
        end
        %
        iz = C.H.gc>0;
        h  = C.H(iz,:);
        tx1 = C.qpscale*(h.mx1/C.pscale);
        ty1 = C.qpscale*(h.my1/C.pscale);
        tx2 = C.qpscale*(h.mx2/C.pscale);
        ty2 = C.qpscale*(h.my2/C.pscale);
        
        rx1 = C.qpscale*(h.px/C.pscale);
        ry1 = C.qpscale*(h.py/C.pscale);
        rx2 = rx1+C.qpscale*(C.fwidth/C.pscale);
        ry2 = ry1+C.qpscale*(C.fheight/C.pscale);
        %
        for i=1:numel(tx1)
            rect2lines(tx1(i),ty1(i),tx2(i),ty2(i),'y-');
        end
        %
        for i=1:numel(tx1)
            rect2lines(rx1(i),ry1(i),rx2(i),ry2(i),'r-');
        end
        %        
    end
    %
    title(replace(C.samp,'_','-'));
    hold off
    shg
    %
    xx = get(gca,'XLim')/C.qpscale;
    yy = get(gca,'YLim')/C.qpscale;
    pp = get(gca,'Position');
    %
    C.axis = [xx,yy];
    C.posn = pp;
    %
end


function p = rect2lines(x1,y1,x2,y2,t)
%%----------------------------------------------------
%% draw a rectangle with the diagonal (x1,y1)-(x2,y2)
%%----------------------------------------------------
    %
    x = [x1,x2,x2,x1,x1];
    y = [y1,y1,y2,y2,y1];
    p = plot(x,y,t,'LineWidth',0.5);
    %
end