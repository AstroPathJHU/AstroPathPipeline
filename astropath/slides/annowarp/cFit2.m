function [fitres] = cFit2(x, y, z)
    %
    [xx,yy,zz] = prepareSurfaceData( x, y, z );
    opts = fitoptions( 'Method', 'LinearLeastSquares','Robust','on');
    [fitres,gof] = fit([xx,yy],zz,fittype('poly11'), opts);
    %
end
