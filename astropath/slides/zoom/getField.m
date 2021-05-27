function img = getField(C,n)
%%-------------------------------------
%% Read the next field's image layers.
%% Set img=[] if the tiff file is missing
%%
%% 2020-06-25   Alex Szalay
%%-------------------------------------
    %
    f = fullfile(C.tiffpath,replace(C.H.file{n},'.im3',C.tiffext));
    img = [];
    try
        img = fastTiff(f,C.fmax);
        %fprintf('%d,%s\n',n,C.H.file{n});
    catch
        s = sprintf('ERROR: cannot read %s',f);
        logMsg(C,s,1);
        return
    end
    %
end