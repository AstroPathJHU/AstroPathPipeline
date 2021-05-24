function M = addModels(C,O)
%%---------------------------------------------------
%% Generate additional columns from the modelFit.
%% M.mdx, M.mdy: analytic model fit
%% M.rx, M.ry: residuals after the model subtraction
%%---------------------------------------------------
    %
    logMsg(C,'addModels');
    %
    M = C.Y.M;
    %
    bx = C.Y.R.bx;    
    M.mdx = bx(1)*double(M.xx)+bx(2)*double(M.yy)+...
        bx(3)+bx(4)*double(M.n)+bx(5)*double(M.m);
    %
    by = C.Y.R.by;
    M.mdy = by(1)*double(M.xx)+by(2)*double(M.yy)+...
        by(3)+by(4)*double(M.n)+by(5)*double(M.m);
    %
    M.rx = M.dx-M.mdx;
    M.ry = M.dy-M.mdy;
    %
end