function w = getWarp(C,sz)
%%---------------------------------------------------
%% Read the warp map from a disk file.
%% sz is the tile size in ipix pixels
%%
%% 2020-05-19   Alex Szalay
%%---------------------------------------------------
    %
    w =[];
    fname = sprintf('%s-warp-%d.csv',C.samp,sz);
    try
        w = readtable(fname);
    catch
        fprintf('%s not found\n',fname);
        return
    end
    %
end