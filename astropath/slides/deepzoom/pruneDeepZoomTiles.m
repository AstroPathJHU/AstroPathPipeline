function [dgood] = pruneDeepZoomTiles(C,layer)
%%---------------------------------------------------------
%% remove the blank tiles from the DeepZoom directories
%% 2020-12-07   Alex Szalay
%%---------------------------------------------------------
    %------------------------
    % get the directories
    %------------------------
    lyr = sprintf('%d',layer);
    dst = [C.zdest,'\L',lyr,'_files\**\*.png'];
    %-------------------------------------
    % get the list of all the tile files
    %-------------------------------------
    d = dir([dst]);
    d = struct2table(d);
    d.fname = strcat(d.folder,'\',d.name);
    %----------------------
    % find the blank tiles
    %----------------------
    ix = (d.bytes==162);
    dempty = d(ix,:);
    dgood  = d(~ix,:);
    msg = sprintf('Prune Layer(%d): %d non-empty files out of %d',...
            layer, numel(dgood.name),numel(d.name));      
    logMsg(C,msg);
    %-------------------------
    % delete the empty files
    %-------------------------
    for i=1:numel(dempty.name)
        if (C.opt==0)
            try
                delete(dempty.fname{i});
            catch
                msg = sprintf(['WARNING: Cannot find %s ',...
                    'to delete in Layer(%d)\n'],dempty.fname{i},layer);
                logMsg(C,msg);
            end
        end
    end
    if (C.opt>0)
        msg = sprintf('Layer(%d): will delete %d files kept out of %d',...
            layer, numel(dempty.name),numel(d.name));
        logMsg(C,msg);
    end
    %------------------------------
    % get the absolute zoom number
    %------------------------------
    dgood.tlevel = cellfun(@str2num,...
        replace(dgood.folder,[C.zdest,'\L',lyr,'_files\'],''));
    maxlevel=max(dgood.tlevel);
    dgood.zoom = dgood.tlevel +9-maxlevel;    
    %
end