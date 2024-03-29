%% processseg
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% 
%% --------------------------------------------------------------
%%
function processseg(main)
%
try
    tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
        'ReadVariableNames', true);
    %
    tbl2 = readtable([main, '\AstropathConfig.csv'], 'Delimiter' , ',',...
        'ReadVariableNames', true);
    %
catch
    pause(10)
    tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
        'ReadVariableNames', true);
    %
    tbl2 = readtable([main, '\AstropathConfig.csv'], 'Delimiter' , ',',...
        'ReadVariableNames', true);
    %
end
%
for i1 = 1:height(tbl)
    %
    % Clinical_Specimen folder
    %
    td = tbl(i1,:);
    wd = ['\\', td.Dpath{1},'\', td.Dname{1}];
    project = td.Project;
    %
    gostr = tbl2(tbl2.Project == project,'segmaps');
    gostr = table2array(gostr);
    gostr = gostr{1};
    if strcmpi(gostr,'No')
        continue
    end
    %
    % get specimen names for the CS
    %
    samplenames = find_specimens(wd);
    %
    % cycle through and create flatws
    % 
    for i2 = 1:length(samplenames)
        sname = samplenames{i2};
        disp(['Started ',sname, ' Slide Number: ',num2str(i2)]);   
        %
        % get the BatchID 
        %
        try
            [~, ~, BatchID] = getscan(wd, sname);
            MergeConfig = [wd,'\Batch\MergeConfig_',BatchID,'.xlsx'];
        catch
            fprintf(['"',sname,'" is not a valid clinical specimen folder \n']);
            continue
        end
        %
        % determine when MSI folder was created as Scan date tracker
        %
        startseg(wd,sname, MergeConfig)
        disp(['Completed ',sname, ' Slide Number: ',num2str(i2)]);    
    end
end
end
%% startseg
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% 
%% --------------------------------------------------------------------
%%
function startseg(wd, sname, MergeConfig)
%
wd1 = [wd,'\',sname,'\im3\flatw'];
fd = [wd,'\',sname,'\inform_data\Component_Tiffs'];
wd2 = [wd,'\',sname,'\inform_data\Phenotyped\Results\Tables'];
%
if exist(fd, 'dir') && exist(wd2,'dir')
    fnames = dir([fd,'\*_w_seg.tif']);
    t1 = min([fnames(:).datenum]);
    fnames2 = dir([wd1, '\*.im3']);
    fnames3 = dir([wd2, '\*.csv']);
    t3 = max([fnames3(:).datenum]);
    onedayago = datenum(datetime() - 1);
    %
    if ~isempty(t3) && ...
            (length(fnames) ~= length(fnames2)|| t3 > t1) %&& t2 <= onedayago
        %
        % start the parpool if it is not open;
        % attempt to open with local at max cores, if that does not work attempt
        % to open with BG1 profile, otherwise parfor should open with default
        %
        if isempty(gcp('nocreate'))
            try
                numcores = feature('numcores');
                if numcores > 12
                    numcores = numcores/4;
                end
                evalc('parpool("local",numcores)');
            catch
                try
                    numcores = feature('numcores');
                    if numcores > 12
                        numcores = numcores/4;
                    end
                    evalc('parpool("BG1",numcores)');
                catch
                end
            end
        end
        %
        parfor i5 = 1:length(fnames)
            fid = fullfile(fnames(i5).folder,fnames(i5).name);
            delete(fid)
        end
        if ~isempty(gcp('nocreate'))
            poolobj = gcp('nocreate');
            delete(poolobj);
        end
        %
        GetaSeg(wd, sname, MergeConfig);
        %
        GetnoSeg(wd, sname, MergeConfig)
        %
    end
end
%
end