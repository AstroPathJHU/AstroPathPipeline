function [xf] = xform1(img,MAX)
%%----------------------------------------------------
%% does a asinh remapping of the image intensities
%% it should really be done on the average intensity
%% of all the bands
%%----------------------------------------------------
    %
    SIGMA = 25.0;
    %
    sc  = 255/asinh(MAX/SIGMA);
    xf  = sc*asinh(double(img/SIGMA));
    %
end