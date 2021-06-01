function C = runZoom(root1,samp,varargin)
%%-------------------------------------------------
%% Create the mosaic images from a given sample.
%% layer is the array of layers to process
%%
%% Alex Szalay, 2019-04-05
%%-------------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    layer=(1:8);
    %------------------------------
    % build the config for logging
    %------------------------------
    C = getConfig(root1,samp,'runzoom');
    logMsg(C,'runZoom started',1);
    %------------------------------
    % initialize C for the work
    %------------------------------
    C.opt = opt;
    C = prepZoom(C,layer);
    if (C.bits~=8 & C.bits~= 16)
        fprintf('Unknown bit depth %d\n', C.bits);
        return
    end
    if (C.opt==-1)
        return
    end
    %
    %makeLumi(C);
    %-----------------
    % get the images
    %-----------------
    %     C = getLayersRaw(C);
    %     return
    %    
    %----------------------------------
    % do the stitching into 16K tiles
    %----------------------------------
    makeZoom(C);
    if (C.opt==-2)
        return
    end    
    %
    % delete the buf directory, if there
    %
    f = fullfile(C.zoompath,'buf');
    if (exist(f)==7)
        delete([f,'\*.*']);
        rmdir(f);
    end
    %
    %----------------------------------
    % do the stitching into 16K tiles
    %----------------------------------
    mergeBigTiles(C);
    if (C.opt==-2)
        return
    end        
    %
    return
    %return
    %
    mergeZoom(C);
    %
    splitZoom(C);
    %
    pool = gcp('nocreate');
    if (~isempty(pool))
        delete(pool);
    end
    %
    % delete the big directory
    %{
    f = [C.zoompath,'big'];
    delete([f,'\*.*']);
    rmdir(f);
    %}
    writeZoomList(C);
    %
    s = sprintf('runZoom finished in %f sec',etime(clock(),t0));
    logMsg(C.samp,s,1);
    %
end


