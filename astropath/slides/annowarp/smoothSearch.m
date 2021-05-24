function f = smoothSearch(C)
%%-------------------------------------------------------
%% Take the crosscorrelation matrix C
%% A and B are the two images, WW is the smoothing length
%% (-LL,LL) is the grid range, and
%% x0,y0 is the relative shift between the centers. 
%% The function fits a bicubic spline to teh grids, then does
%% a gradient search on the interpolated function for the
%% minimum.  Returns a struct containing the final shift, 
%% and the the grid points for debugging.
%%
%% Alex Szalay, Baltimore, 2018-07-27
%%--------------------------------------------------
    %    
    x0 = (size(C,1)-1)/2;
    y0 = (size(C,2)-1)/2;
    [f.X,f.Y] = meshgrid((-x0:x0),(-y0:y0));
    f.x0=0;
    f.y0=0;
    %
    f.V = 100*log(1+double(C));
    %------------------------------------
    % fit cubic spline to the cost fn
    %------------------------------------
    f.ft = fitS2(f.X,f.Y,f.V);
    %-----------------------------------------
    % search the fitted function for maximum
    % over a higher resolution grid
    %-----------------------------------------
    x = (-5:0.1:5);
    y = (-5:0.1:5);
    [X,Y] = meshgrid(x,y);
    W = feval(f.ft,[X(:),Y(:)]);
    W = reshape(W,numel(x),numel(y));
    [ix,iy] = find(W==max(W(:)));
    %
    f.dx = x(ix);
    f.dy = y(iy);
    %
end


function [fitres] = fitS2(x, y, v)
%%------------------------------------------
%%  Create a cubic spline fit
%%  Alex Szalay, Baltimore, 2018-07-27
%%------------------------------------------
    %
    [xData, yData, zData] = prepareSurfaceData(x,y,v);
    ft = 'cubicinterp';
    [fitres, gof] = fit([xData,yData],zData,ft,'Normalize','on' );
    %
end




