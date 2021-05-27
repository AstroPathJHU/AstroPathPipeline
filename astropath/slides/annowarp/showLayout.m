function [C] = showLayout(C,test)
%%-------------------------------------------------
%% show the layout of the high-powered fields
%%  test=0 just shows the image
%%  test=1 shows the HPF outlines
%%  test=2 shows the outlines with the scanning pattern
%%  test=3 qimg with rescale
%%  test=4 aimg
%%  test=5 aimg
%%  test=6 aimg
%%
%% Alex Szalay, Baltimore, 2018-11-14
%%-------------------------------------------------
    %
    figure(test)
    if (test==1)
        img = 0.7*imresize(C.Q.img,C.ppscale);
        fprintf('Color QPTIFF scaled up by 2 (%d,%d)\n',...
            size(img,1),size(img,2));
    elseif (test==2)
        img = 2*imresize(C.qimg,1/C.iqscale);
        imshow(img);
        fprintf('C.qimg at the original QP scale (%d,%d)\n',...
            size(img,1),size(img,2));
    elseif (test==3)
        img = 2*imresize(C.qimg,1.0);
        imshow(img);
        fprintf('C.qimg rescaled to AP scale (%d,%d)\n',...
            size(img,1),size(img,2));
    elseif (test==4 | test==5 | test==6)
        img = imresize(4*C.aimg,1.0);
        fprintf('C.aimg (%d,%d)\n',size(img,1),size(img,2));
    else
        return
    end
    %
    imshow(img);
    set (gca,'Ydir','reverse');
    hold on
    showOutlines(C,test,1);     
    hold off
    shg
    %
end


function showOutlines(C, test, flag)
    %----------------------------------------
    % field offsets and sizes all in microns
    %----------------------------------------
    pw = C.fwidth/C.pscale;   % size in microns
    ph = C.fheight/C.pscale;  % size in microns
    %
    if (test==1)
        sc  = C.qpscale*C.ppscale;
    elseif (test==2)
        sc = C.apscale;
    else
        % convert micron to pixels then resize to actual
        sc = C.pscale/C.ppscale;
    end
    %
    if (flag==0)
        tx1 = sc*(C.H.px/C.pscale);
        ty1 = sc*(C.H.py/C.pscale);
        tx2 = tx1 + sc*pw;
        ty2 = ty1 + sc*ph;
    elseif (flag==1)
        tx1 = sc*(C.H.mx1/C.pscale);
        ty1 = sc*(C.H.my1/C.pscale);    
        tx2 = sc*(C.H.mx2/C.pscale);
        ty2 = sc*(C.H.my2/C.pscale);
    end
    %-------------------------
    % show the field outlines
    %-------------------------
    N = numel(tx1);
    for i=1:N
        t = '-';
        rect2lines(tx1(i),ty1(i),tx2(i),ty2(i),t);
    end
    %
end



function p = rect2lines(x1,y1,x2,y2,t)
%%----------------------------------------------------
%% draw a rectangle with the diagonal (x1,y1)-(x2,y2)
%%----------------------------------------------------
    %
    x = [x1,x2,x2,x1,x1];
    y = [y1,y1,y2,y2,y1];
    p = plot(x,y,t,'LineWidth',0.5,'Color',[0.5,0.5,0.5]);
    %
end

