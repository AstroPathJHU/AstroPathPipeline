function v = modelMap(x,y,R,F)
%%---------------------------------------------------
%% Evaluate the fit function at the (x,y) positions
%%
%% 2020-06-06   Alex Szalay
%%---------------------------------------------------
    n = floor(x/R.tx);
    m = floor(y/R.ty);
    X = n*R.tx;
    Y = m*R.ty;
    xx = x-X;
    yy = y-Y;
    %
    v.dx = R.bx(1)*xx+R.bx(2)*yy+R.bx(3)+R.bx(4)*n+R.bx(5)*m;
    v.dy = R.by(1)*xx+R.by(2)*yy+R.by(3)+R.by(4)*n+R.by(5)*m;
    %
end