function V = warpAnnotations(C)
%%-----------------------------------------------------
%% Apply the computed warp model to the annotations
%% and recast the vertices in AP pixel units
%%
%% 2020-07-13   Alex Szalay
%%-----------------------------------------------------
    %
    logMsg(C,'warpAnnotations');
    %
    R = C.Y.R;
    F = C.Y.F;
    %
    V  = C.PV;
    xx = V.x*C.iqscale;
    yy = V.y*C.iqscale;
    vv = modelMap(xx,yy,R,F);
    %------------------------------------------------
    % apply the warp and scale them to the AP scale
    %------------------------------------------------
    V.wx = round(C.ppscale*(xx+vv.dx));
    V.wy = round(C.ppscale*(yy+vv.dy-C.qshifty));
    %
end