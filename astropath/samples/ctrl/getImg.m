function T = getimg(fname)
%%-------------------------------------
%% read the component_data.tif file and return
%% a struct with all the information
%%
%% Alex Szalay, Baltimore, 2018-06-10
%%--------------------------------------
    %
    T.info  = imfinfo(fname);
    %fprintf('%s\n',fname);
    N = numel(T.info);
    for k = 1:9
        T.img{k} = imread(fname, k, 'Info', T.info);
    end
end


