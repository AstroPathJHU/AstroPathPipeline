function T = getZoomList(C)
    %
    %------------------------
    % get the directories
    %------------------------
    dst = [C.zdest,'\**\*.png'];
    %-------------------------------------
    % get the list of all the tile files
    %-------------------------------------
    d = dir([dst]);
    d = struct2table(d);
    d.fname = strcat(d.folder,'\',d.name);
    d.tag = strcat(replace(replace(d.folder,[C.zdest,'\L'],''),...
        'files\Z',''),'_',replace(d.name,'.png',''));
    %
    cc = split(d.tag,'_');
    %
    for i=1:numel(d.name)
        d.sample{i}=C.samp;
    end
    d.zoom   = cellfun(@str2num,cc(:,2));
    d.marker = cellfun(@str2num,cc(:,1));
    d.x      = cellfun(@str2num,cc(:,3));
    d.y      = cellfun(@str2num,cc(:,4));
    %------------------------------------------------------
    % extract columns and sort in the correct index order
    %------------------------------------------------------
    T = d(:,{'sample','zoom','marker','x','y','fname'});
    T.Properties.VariableNames{6}='name';
    T = sortrows(T,{'zoom','marker','x','y'});
    %
end