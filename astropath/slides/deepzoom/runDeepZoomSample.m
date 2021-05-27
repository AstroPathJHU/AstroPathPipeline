function C=runDeepZoomSample(root,samp,varargin)
%%---------------------------------------------
%% run the DeepZoom image tiling over the 
%% WSI image layers
%%
%% 2020-12-07   Alex Szalay
%%---------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %------------------------------
    % build the config for logging
    %------------------------------
    C = getConfig(root,samp,'deepzoom');
    logMsg(C,'runDeepZoomSample started',1);
    %------------------------------
    % initialize C for the work
    %------------------------------
    C.opt = opt;
    C.zoomroot = '\\BKI01\c$\data\data11\zoom\';
    C.deepzoom = '\\BKI01\c$\data\data04\deepzoom\';
    C.layers=(1:8);
    %
    %--------------------------------------------------
    % test zoom directory, if it does not exist
    %--------------------------------------------------
    C.zpath = fullfile(C.zoomroot,sprintf('Project%02d',C.project),samp);
    if (exist(C.zpath)==0)
        logMsg(C,'ERROR: zoom directory not found',1);
        C.err=1;
        return
    end
    %-------------------------------------------
    % make zoom directory, if it does not exist
    %-------------------------------------------
    C.zdest = fullfile(C.deepzoom,sprintf('Project%02d',C.project),samp);
    if (exist(C.zdest)==0)
        mkdir(C.zdest);
    end
    %
    %--------------------------------------------------
    % check for the WSI files, exit if missing or none
    %--------------------------------------------------
    d=struct2table(dir(fullfile(C.zpath,'wsi\*.png')));
    if (numel(d.name)==0)
        C.err=1;
        msg = 'ERROR: wsi images missing';
        logMsg(C,msg,1);
        return;
    elseif (numel(d.name)~=8)
        C.err=1;
        msg = sprintf('ERROR: only %d wsi images found',numel(d.name));
        logMsg(C,msg,1);
        return;
    end        
    if (C.opt==-1)
        return
    end
    %    
    runVIPSLoop(C);
    %
    if (C.err==1)
        logMsg(C,'ERROR: VIPS retry count exceeded',1);
        return
    end
    %---------------------------------
    % get the imagelist for loading
    % and write it to the sample root
    %---------------------------------
    if (C.opt>=0)
        T = getZoomList(C);
        fname = fullfile(C.zdest,'zoomlist.csv');
        writetable(T,fname);
    end
    %
    logMsg(C,'runDeepZoomSample finished',1);
    %
end

