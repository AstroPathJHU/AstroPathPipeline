function C = quantizeGrid(C,n)
%%-------------------------------------------------------------
%% Build a discrete grid of the images in the n-th partition
%% Alex Szalay, Baltimore, 2018-08-09
%%-------------------------------------------------------------
    %
    logMsg(C,mfilename);
    %
    % build a hi-res histogram of the coordinates
    % quantize by 100 pixels, threshold and then
    % build a label array with the group ids
    %
    g = (C.H.gc == n);
    X = C.H.px(g);
    Y = C.H.py(g);
    %
    dw = 100;
    %
    % start with the x-coordinate
    %
    x1 = dw*(floor(min(X)/dw)-1);
    x2 = dw*(floor(max(X)/dw)+1);
    ex = (x1:dw:x2);
    hn = histcounts(X,ex);
    nx = bwlabel(hn>0);
    %
    ix = floor((X-x1)/dw+1);
    gx = nx(ix);
    C.R.gx(g) = gx;
    %
    % now do the y-coordinates
    %
    y1 = dw*(floor(min(Y)/dw));
    y2 = dw*(floor(max(Y)/dw)+1);
    ey = (y1:dw:y2);
    hn = histcounts(Y,ey);
    ny = bwlabel(hn>0);
    %
    iy = floor((Y-y1)/dw+1);
    gy = ny(iy);
    C.R.gy(g) = gy;
    %
    % for each gx value compute the mean of X over the grid
    %
    NX = max(gx);
    xx = grpstats(X,gx);
    for i=2:NX
        mx(i) = floor(0.5*(xx(i-1)+C.fwidth+xx(i))); 
        dx(i) = floor(0.5*(xx(i-1)+C.fwidth-xx(i)));
    end
    %
    % adjust first and last, by the mean of the half-width
    %
    DX = floor(mean(dx));
    mx(1) = xx(1)+DX;
    mx(NX+1) = xx(NX)+C.fwidth-DX;
    %
    yy = grpstats(Y,gy);
    %
    % for each gx value compute the mean of X over the grid
    %
    NY = max(gy);
    yy = grpstats(Y,gy);
    for i=2:NY
        my(i) = floor(0.5*(yy(i-1)+C.fheight+yy(i))); 
        dy(i) = floor(0.5*(yy(i-1)+C.fheight-yy(i)));
    end
    %
    % adjust first and last, by the mean of the half-width
    %
    DY = floor(mean(dy));
    my(1) = yy(1)+DY;
    my(NY+1) = yy(NY)+C.fheight-DY;
    %
    for i=1:NX
        gi = (gx==i);
        mx1(gi) = floor(mx(i));
        mx2(gi) = floor(mx(i+1));
    end
    %
    for i=1:NY
        gi = (gy==i);
        my1(gi) = floor(my(i));
        my2(gi) = floor(my(i+1));
    end
    %
    % write it back into place for the whole partition
    %
    C.H.mx1(g) = mx1;
    C.H.mx2(g) = mx2;
    C.H.my1(g) = my1;
    C.H.my2(g) = my2;
    %
end
