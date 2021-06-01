function a = getLumiSum(samp)
%%----------------------------------------------------
%% use the images with the sum of the fluxes to build
%% a histogram of the log fluxes at a fixed resolution
%% so that we can compare the different slides
%%----------------------------------------------------
    %
    ff = ['Y:\lumi\',samp,'\*.tif'];
    d  = dir(ff);
    %
    a = double(imread([d(1).folder,'\',d(1).name]));
    for n=2:numel(d)
        a = a + double(imread([d(n).folder,'\',d(n).name]));
    end
    a = a/numel(d);
    %    
end