%% getinformfiles
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% Will move all inform files from a \tmp_inform_data folder to the proper
%%% specimen and track how many files exist, dates, and how many files
%%% should exist
%% --------------------------------------------------------------
%%
function [insnum, expectedinform, infm, infmd, trackinform,...
    iffdloc, iffd, expectedTablesnum] ...
    = getinformfiles(sname, actualim3num_internal, tmpfd, informpath)
    %
    % some file tracking vectors
    % insum tracks number of actual inform files exist;
    %
    insnum = zeros(length(tmpfd),1);
    %
    % expectedinform tracks number of files expected from flatw output and
    % any errors that may show up in the Batch files
    %
    expectedinform = zeros(length(tmpfd),1);
    % 
    infm = cell(length(tmpfd),1);
    infmd = cell(length(tmpfd),1);
    iffdloc = zeros(length(tmpfd),1);
    dt = cell(length(tmpfd),1);
    trackinform = 0;
    for i2 = 1:length(tmpfd)
        tmpname = tmpfd(i2).name;
        %
        % If new inform files are in tmp_inform_data folder for that specimen
        % then move them into the proper folder, using Batch.log as finished
        % designation
        %
        % This will remove other inform_files if they exist
        %
        % First find the inform files that are generated in tmp folders; to
        % do this we must search subdirectories in each ABxx folder
        %
        tmp2path = [tmpfd(i2).folder,'\',tmpname];
        tmp2 = dir(tmp2path);
        tmp2 = tmp2(3:end);
        ii = [tmp2.isdir];
        tmp2 = tmp2(ii);
        %
        % loop through each *\ABxx\'subfolder'
        %{
        for i3 = 1:length(tmp2)
            %
            % numeric folder paths are strings relegating the subdir
            % paths
            %
            numericfdspath = [tmp2(i3).folder,'\',tmp2(i3).name];
            %
            % Search for *\Batch.log file to indicate that the inform 
            % is finished
            %
            Batch = dir([numericfdspath,'\Batch.*']);
            if ~isempty(Batch)
                %
                % now check the files and find out if any correspond to
                % this case
                %
                cfiles = dir([numericfdspath,'\',sname,'_*']);
                %cfiles = cfiles(3:end);
                if ~isempty(cfiles)
                    %
                    % transfer those files that are for this case
                    %
                    %
                    % start the parpool if it is not open;
                    % attempt to open with local at max cores, if that does not work attempt
                    % to open with BG1 profile, otherwise parfor should open with default
                    %
                    if isempty(gcp('nocreate'))
                        try
                            numcores = feature('numcores');
                            if numcores > 10
                                numcores = 8;
                            end
                            evalc('parpool("local",numcores)');
                        catch
                            try
                                numcores = feature('numcores');
                                if numcores > 10
                                    numcores = 8;
                                end
                                evalc('parpool("BG1",numcores)');
                            catch
                            end
                        end
                    end
                    tmp3path = [numericfdspath,'\',sname,'_*'];
                    des1 = [informpath,'\Component_Tiffs'];
                    sor = [tmp3path,'component_data.tif'];
                    [comps] = transferfls(sor,des1);
                    %
                    des = [informpath,'\Phenotyped\',tmpname];
                    %
                    sor = [tmp3path,'.txt'];
                    [~] = transferfls(sor,des);
                    %
                    sor = [tmp3path,'binary_seg_maps.tif'];
                    [~] = transferfls(sor,des);
                    %
                    sor = [numericfdspath,'\Batch.*'];
                    %
                    ii = dir(fullfile(des,'Batch.*'));
                    if ~isempty(ii)
                        delete(fullfile(ii.folder,ii.name))
                    end
                    copyfile(sor,des);
                    %
                    if comps
                        ii = dir(fullfile(des1,'Batch.*'));
                        if ~isempty(ii)
                            delete(fullfile(ii.folder,ii.name))
                        end
                        copyfile(sor,des1);
                    end
                    %
                    sor = [numericfdspath,'\*.ifp'];
                    ii = dir(fullfile(des,'*.ifp'));
                    if ~isempty(ii)
                       delete(fullfile(ii.folder,ii.name))
                    end
                    copyfile(sor,des);
                    if comps
                        copyfile(sor,des1);
                    end
                end
            end
        end
        %}
        % now check if that inform folder exists in current specimen and 
        % create trackers
        %
        iffd = dir([informpath,'\Phenotyped\']);
        iffd = iffd(3:end);
        [x,y] = ismember(tmpname,{iffd(:).name});
        iffdloc(i2) = y;
        %
        if x == 1
            ins = iffd(y);
            inspath = [ins.folder,'\',ins.name];
            %
            % get number of files in Specified ABxx folder
            %
            insnames = dir([inspath,'\*seg_data.txt']);
            insnum(i2) = length(insnames);
            if ~isempty(insnames)
                trackinform = trackinform + 1;
                %
                % get number of files inForm had an error on to calculate
                % expected number of files
                %
                Bf = dir([inspath,'\Batch.*']);
                expectedinform(i2) = getInFormErrors(sname, Bf, actualim3num_internal);
                %
                % make the number of files string
                %
                infm{i2} = [num2str(insnum(i2)),'of',num2str(expectedinform(i2))];
                %
                % find the most recent transfer date
                %
                [~,idx] = max([insnames(:).datenum]);
                infmd{i2} = insnames(idx).date(1:11);
                dt{i2} = insnames(idx).date(1:11);
            end
            %
            transferComponents(inspath);
            %
        end
    end
    if ~isempty(gcp('nocreate'))
        poolobj = gcp('nocreate');
        delete(poolobj);
    end
    expectedTablesnum = actualim3num_internal;
    if ~isempty(expectedinform)
        expectedTablesnum = min(expectedinform);
    end
    
end