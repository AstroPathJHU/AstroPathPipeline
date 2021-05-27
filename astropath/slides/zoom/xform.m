function [xf] = xform(img)
%%----------------------------------------------------
%% does a asinh remapping of the image intensities
%% it should really be done on the average intensity
%% of all the bands
%%----------------------------------------------------
    %
    MAX   = 80;
    SIGMA = 25.0;
    %
    sc  = 255/asinh(MAX/SIGMA);
    xf  = sc*asinh(img/SIGMA);
    %
end