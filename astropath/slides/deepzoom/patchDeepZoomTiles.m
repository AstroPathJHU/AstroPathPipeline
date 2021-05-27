function dz = patchDeepZoomTiles(C,layer)
%%----------------------------------------------------
%% Renumber the zoom levels, so that Z9 is the raw pixels, 
%% and fill in the lowest missing zoom levels with scaled images
%%
%% 2020-12-07   Alex Szalay
%%----------------------------------------------------
    %
	msg = sprintf('Relabeling Layer(%d) zooms',layer);
	logMsg(C,msg);
	%
    lyr = sprintf('L%d_files',layer);
    dz = struct2table(dir(fullfile(C.zdest,lyr)));
    dz = dz(~ismember(dz.name,{'.','..'}),:);
    dz.tlevel = cellfun(@str2num,dz.name);
    %
    maxlevel = max(dz.tlevel);
    dz.zoom = dz.tlevel +9-maxlevel;
    %
    for i=1:numel(dz.name)
        dz.newfolder{i} = [dz.folder{i},'\Z',sprintf('%d',dz.zoom(i))];
        dz.oldfolder{i} = [dz.folder{i},'\',dz.name{i}];
        movefile(dz.oldfolder{i}, dz.newfolder{i});
    end
    %-------------------------------------
	% fill in the missing small images
	%-------------------------------------
	fillZoomFactors(C,layer);
	%
end
