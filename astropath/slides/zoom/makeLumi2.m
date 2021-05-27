function F = makeLumi(C)
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
        [img, q] = getLumi(C,n,1);
        F{n} = img;
    end
    %
    s = 'makeLumi finished';
    logMsg(C,s);
    %
end


function [img,lm,q] = getLumi(C,n,flag)
%%-------------------------------
%% process a single field
%%-------------------------------
    %
    f = fullfile(C.tiffpath,replace(C.H.file{n},'.im3',C.tiffext));
    g = fullfile(C.lumipath,replace(C.H.file{n},'.im3','-lumi.tif'));
    %
    % return if it exists
    %
    if (exist(g)~=0)
        return
    end
    %
    try
        %
        % read all the images
        %
        for i=1:8
            img{i} = imread(f,i);
        end
        %
        % add all the markers, but not the autofluorescence
        %
        lm = float(img{1});
        for i=2:7
            lm = lm+img{i};
        end
        %
        q = prctile(b(:),[99]);
        %
        % make sure there are no zero flux pixels
        %       
    catch
        msg = sprintf('ERROR: cannot find %s',f);
        logMsg(C,msg,1);
        returna
    end
    %
    img(img(:)==0)=0.1;
    if (flag==1)
        tiffwrite(img,g);
    end
    %
end
