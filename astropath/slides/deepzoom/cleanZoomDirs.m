function d=cleanZoomDirs(proj)
%%-----------------------------------------------------
%% clean up the directory structures of the WSI images
%% 2020-12-07   Alex Szalay
%%-----------------------------------------------------
    %
    zpath = '\\bki01\c$\data\data19\zoom\';
    path = fullfile(zpath,sprintf('Project%d',proj));
    %
    d = dir(path);
    d = struct2table(d);
    d = d(~ismember(d.name,{'.','..'}),:);
    %--------------------------------
    % go through the samples
    %--------------------------------
    for i=1:numel(d.name)
        dname = fullfile(d.folder{i},d.name{i});
        fprintf('%s\n', dname);        
        dd = struct2table(dir(dname));
        dd = dd(~ismember(dd.name,{'.','..','wsi'}),:);
        if (numel(dd)==0)
            continue
        end
        %{
        %---------------------
        % deal with big
        %---------------------
        bname = fullfile(dd.folder{1},'big');
        %fprintf('..%s\n', bname);       
        big = struct2table(dir(bname));
        if (numel(big.name)>0)
            big = big(~ismember(d.name,{'.','..','Thumbs.db'}),:);        
            fprintf('..%s  [%d]\n', bname, numel(big.name));
        end
        continue
        %
        for j=1:numel(big.name)
            pname = fullfile(big.folder{j},big.name{j});
            %fprintf('....%s\n',pname);
            delete(pname);
        end
        %}
        %---------------------
        % remove the subdirectories
        %---------------------
        for j=1:numel(dd.name)
            pname = fullfile(dd.folder{j},dd.name{j});
            %fprintf('%s\n',pname);
            if(~strcmp(dd.name{j},'big'))
                rmdir(pname);
            else
                cleanBigFiles(pname);
                rmdir(pname);
            end
        end
        %
    end
end


function cleanBigFiles(bname)
    %----------------
    % deal with big
    %----------------
    fprintf('...%s\n',bname);
    big = struct2table(dir(bname));
    %
    big = big(~ismember(big.name,{'.','..','Thumbs.db'}),:);
    fprintf('...%s  [%d]\n', bname, numel(big.name));
    %
    for j=1:numel(big.name);
       fname = fullfile(big.folder{j},big.name{j});
       delete(fname);
    end
    %
end