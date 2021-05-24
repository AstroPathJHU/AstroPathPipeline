function [fitresult, gof] = cFit1(x,y,z)
    [xData, yData, zData] = prepareSurfaceData( x, y, z );
    ft = fittype( 'poly11' );
    [fitresult, gof] = fit( [xData, yData], zData, ft );
end
