function Z = xcregister(C,sz,x,y)
%%---------------------------------------------------------------
%% Register the two images to the nearest pixel using xcorr.
%% Cutouts are at (x,y), size sz pixels from both images.  
%%  Z.aimg: Astropath cutout
%%  Z.qimg: QPTiff cutout
%%  Z.c: cross correlation
%%  Z.dx, Z.dy: shift of the Z.q image for best match
%%  Example:
%%      imshowpair(Z.b,imtranslate(Z.q,[Z.dx,Z.dy]);
%%
%% 2020-05-19   Alex Szalay
%%-------------------------------------------------------------
    %
    Z.x  = x;
    Z.y  = y;
    Z.flag = 0;
    Z.sz = sz;
    %
    % extract first image from our own stitching
    %
    if (y+1-C.qshifty<=0)
        return
    end
%    B = double(C.aimg(y+1-C.qshifty:y+sz-C.qshifty,x+1:x+sz));
    B = double(C.aimg(y+1:y+sz,x+1:x+sz));
    Z.B = B;
    W = B>45;
    %--------------------------
    % get the mean intensity
    %--------------------------
    Z.mi = mean(double(W(:)));
    Z.mx = max(B(:));
    Z.mm = Z.mx-min(Z.B(:));
    %-------------------------------
    % quit if too many empty pixels
    % or not enough dynamic range
    %-------------------------------
    if (Z.mi<0.20 | Z.mm<45)
        %fprintf('dark image\n');
        return
    end    
    %
%    Q = double(2*C.qimg(y+1:y+sz,x+1:x+sz));
    Q = double(2*C.qimg(y+1+C.qshifty:y+sz+C.qshifty,x+1:x+sz));
    Z.Q = Q;
    %
    bb = B - imgaussfilt(B,20);
    bb = imgaussfilt(bb,3);
    %
    qq = Q - imgaussfilt(Q,20);
    qq = imgaussfilt(qq,3);
    Z.b = bb;
    Z.q = qq;
    %---------------------------------------------
    % apply the image cross-correlation function
    %---------------------------------------------
    if (max(Q(:))==min(Q(:)))
        %fprintf('<%d,%d> - qptiff is constant\n',x,y);
        return
    end
    %
    c = normxcorr2(qq,bb);
    nc = size(c,1);
    [iy,ix]=find(c==max(c(:)));
    %
    Z.ix = ix;
    Z.iy = iy;
    Z.c = c;
    %
    ww = 11;
    if (iy-ww<=0 | ix-ww<=0 | iy+ww>nc | ix+ww>nc)
        return
    end
    %fprintf('%d %d\n',ix,iy);
    Z.cc = c(iy-ww:iy+ww,ix-ww:ix+ww);  
    %
    f = smoothSearch(Z.cc);
    %
    % get the nearest pixel shift
    %
    Z.dx = ix-sz+f.dx;
    Z.dy = iy-sz+f.dy;
    %
    Z.flag = 1;
    %
end