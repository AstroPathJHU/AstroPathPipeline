function O = test(root,samp)
    C = getMetadata(root,samp);
    C = getImages(C,-1);
    x  = C.PV.x;
    y  = C.PV.y;
    dx = C.PV.wx/C.ppscale-C.PV.x;
    dy = C.PV.wy/C.ppscale-C.PV.y; 
    fx = createFit(x,y,dx);
    fy = createFit(x,y,dy);
    %
    O = {samp,fx.p00,fx.p10,fx.p01,fy.p00,fy.p10,fy.p01};
    %
end


function [fitresult, gof] = createFit(x, y, dy)
%% Fit: 'untitled fit 1'.
    [xData, yData, zData] = prepareSurfaceData( x, y, dy );

    % Set up fittype and options.
    ft = fittype( 'poly11' );
    opts = fitoptions( 'Method','LinearLeastSquares');
    opts.Robust = 'Bisquare';

    % Fit model to data.
    [fitresult, gof] = fit( [xData, yData], zData, ft, opts );
    %{
    % Plot fit with data.
    figure( 'Name', 'untitled fit 1' );
    h = plot( fitresult, [xData, yData], zData );
    legend( h, 'untitled fit 1', 'dy vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
    % Label axes
    xlabel( 'x', 'Interpreter', 'none' );
    ylabel( 'y', 'Interpreter', 'none' );
    zlabel( 'dy', 'Interpreter', 'none' );
    grid on
    %}
end