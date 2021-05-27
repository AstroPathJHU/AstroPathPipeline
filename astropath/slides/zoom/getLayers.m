function F = getLayers(C)
%%---------------------------------------
%% get the relevant layers into memory
%%
%% Alex Szalay, Baltimore, 2019-05-15
%%---------------------------------------
    %
    logMsg(C,'getLayers');
    %    
    F =[];
    if (exist(C.tiffpath)~=7)
        msg = sprintf('ERROR: image path %s missing',path);
        logMsg(msg,1);
        return
    end
    %
    for n=1:numel(C.H.n)
        F{n} = getTiffLayer(C,n);
    end
    %
    s = sprintf('getLayers finished, %d images read',...
        numel(C.H.n));
    logMsg(C,s);
    %
end

function img=getTiffLayer(C,n)
    %
    f = fullfile(C.tiffpath,replace(C.R.file{n},'.im3',C.tiffext));
    g = fullfile(C.lumipath,replace(C.R.file{n},'.im3','-lumi.tif'));
    img = uint8([]);
    try
        for i=1:8
            a(:,:,i)   = imread(f,i);
        end
    catch
        s = sprintf('ERROR: cannot read %s',f);
        logMsg(C.samp,s,1);
        return
    end
    %
    try
        b   = imread(g);       
    catch
        s = sprintf('ERROR: cannot read %s',g);
        logMsg(C.samp,s,1);
        return
    end
    %
    % leave the images as single for the cubic interpolation
    %
    if (C.bits==8)
        for i=1:8
            a(:,:,i) = a(:,:,i)./b.*xform(b);
        end
    elseif (C.bits==16)
        img = 200*a./b.*xform(b);
    end
    %---------------------
    % convert to uint8
    %---------------------
    %
    img = uint8(a);
    %
end
