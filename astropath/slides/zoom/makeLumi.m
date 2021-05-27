function makeLumi(C)
%%---------------------------------------
%% get the relevant layers into memory
%% and compute a luminescence layer as
%% the sum of all 8 layers from the component TIFs.
%% The image will be stored as a 32bit float TIF.
%%
%% Alex Szalay, Baltimore, 2019-05-15
%%---------------------------------------
    %
    t0 = clock();
    logMsg(C,'makeLumi started');
    %
    C.tiffext = sprintf('_component_data.tif');
    C.tiffpath = [C.root '\' C.samp '\inform_data\Component_Tiffs\'];
    F =[];
    if (exist(C.tiffpath)~=7)
        fprintf(['ERROR: image path ' replace(path,'\','/') ' missing']);
        return
    end
    %
    d = dir([C.lumipath,'\*.tif']);
    if (numel(d)==numel(C.H.n))       
        return
    end
    %
    pool = gcp('nocreate');
    if (isempty(pool))
        pool= parpool('local',6);
    end
    %    
    parfor n=1:numel(C.H.n)
        getLumi(C,n);
    end
    %
    s = 'makeLumi finished';
    logMsg(C,s);
    %
end


function getLumi(C,n)
%%-------------------------------
%% process a single field
%%-------------------------------
    %
    f = fullfile(C.tiffpath,replace(C.H.file{n},'.im3',C.tiffext));
    g = fullfile(C.lumipath,replace(C.H.file{n},'.im3','-lumi.tif'));
    %
    % return if it is already there
    %
    if (exist(g)~=0)
        return
    end
    %
    try
        %
        % add all the markers, but not the autofluorescence
        %
        img = imread(f,1);
        for i=2:7
            img = img + imread(f,i);
        end
        %
        % make sure there are no zero flux pixels
        %       
    catch
        msg = sprintf('ERROR: cannot find %s',f);
        logMsg(C,msg,1);
        return
    end
    %
    img(img(:)==0)=0.1;
    tiffwrite(img,g);
    %
end
