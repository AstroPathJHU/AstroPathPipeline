function dd = copyXmlDir()
    %
    oldpath = '\\bki02\e\Clinical_Specimen';
    newpath = '\\bki04\Clinical_Specimen';
    %-----------------------------
    % ftach the xml directories
    %-----------------------------
    d = dir([oldpath,'\*\im3\xml*']);
    d = struct2table(d);
    d.newfolder=replace(d.folder,oldpath,newpath);
    %-------------------------
    % go through the samples
    %-------------------------
    dd=[];
    for i=1:numel(d.bytes)
        src = fullfile(d.folder{i},d.name{i});
        dst = fullfile(d.newfolder{i},d.name{i});
        %
        if (~exist(dst))
            fprintf('mkdir %s\n',dst);
            mkdir(dst);
        end
        %--------------------
        % get all the files
        %--------------------
        dd  = dir(fullfile(src,'*.*'));
        dd  = struct2table(dd);
        dd  = dd(~ismember(dd.name,{'.','..'}),:);        
        fprintf('%s %d\n',src,numel(dd.name));        
        dd.newfolder = replace(dd.folder,src,dst);
        %---------------------
        % do the copy loop
        %---------------------
        for j=1:numel(dd.bytes)
            oldfile = fullfile(src,dd.name{j});
            newfile = fullfile(dst,dd.name{j});
            %fprintf('copyfile %s %s \n', oldfile, newfile);
            copyfile(oldfile, newfile);
        end
    end
    %
end