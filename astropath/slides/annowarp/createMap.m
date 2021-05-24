function T = createMap(C)
%%----------------------------------------------------------------
%% Create a completely model-generated map over the whole image
%% on a regular grid with a fixed spacing bracketing the qptiff cuts,
%% to be used for a linear interpolator.
%% T is the output grid ready to be used with an interpolator
%%
%% 2020-05-21   Alex Szalay
%%----------------------------------------------------------------
    %
    logMsg(C,'createMap');
    %
    R = C.Y.R;
    E = C.Y.E;
    F = C.Y.F;
    %
    %sizex = E.tx*E.nx2;
    %sizey = E.ty*E.ny2;
    %
    gx = ((E.nx1:E.nx2)*R.tx);
    gy = ((E.ny1:E.ny2)*R.ty);
    d = 0.1;
    gridx = unique([0,gx-d,gx+d,(gx(end)+R.tx)]);
    gridy = unique([0,gy-d,gy+d,(gx(end)+R.ty)]);
    %
    [X,Y] = ndgrid(gridx,gridy);
    T = modelMap(X,Y,R,F);
    T.x = X;
    T.y = Y;
    %
end

