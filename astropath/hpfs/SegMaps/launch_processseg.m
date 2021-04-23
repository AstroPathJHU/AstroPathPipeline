%% launch_processseg
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% 
%% --------------------------------------------------------------
%%
function launch_processseg(main)
%
try
    tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
        'ReadVariableNames', true);
catch
    pause(10)
    tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
        'ReadVariableNames', true);
end
%
for i1 = 1:height(tbl)
    %
    % Clinical_Specimen folder
    %
    td = tbl(i1,:);
    wd = ['\\', td.Dpath{1},'\', td.Dname{1}];
    %
    % get specimen names for the CS
    %
    samplenames = find_specimens(wd);
    %
    % cycle through and create flatws
    % 
    for i2 = 1:length(samplenames)
        sname = samplenames{i2};
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
%% getscan 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% get the highest number of a directory given a directory and a specimen
%%% name
%% --------------------------------------------------------------
%%
function [Scanpath, ScanNum, BatchID] = getscan(wd, sname)
%
% get highest scan for a sample
%
Scan = dir([wd,'\',sname,'\im3\Scan*']);
%
sid = {Scan(:).name};
sid = extractAfter(sid,'Scan');
sid = cellfun(@(x)str2double(x),sid,'Uni',0);
sid = cell2mat(sid);
sid = sort(sid,'descend');
ScanNum = sid(1);
%
Scanpath = [wd,'\',sname,'\im3\Scan', num2str(ScanNum)];
BatchID  = [];
fid = fullfile(Scanpath, 'BatchID.txt');
try
    fid = fopen(fid, 'r');
    BatchID = fscanf(fid, '%s');
    fclose(fid);
catch
end

if ~isempty(BatchID) && length(BatchID) == 1
        BatchID = ['0',BatchID];
end
Scanpath = [Scanpath,'\'];
end
%% startseg
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% 
%% --------------------------------------------------------------------
%%
function startseg(wd,sname, MergeConfig)
%
wd1 = [wd,'\',sname,'\inform_data\Phenotyped\Results\Tables'];
fd = [wd,'\',sname,'\inform_data\Component_Tiffs'];
%
if exist(fd, 'dir') && exist(wd1,'dir')
    fnames = dir([fd,'\*_w_seg.tif']);
    t1 = max([fnames(:).datenum]);
    fnames2 = dir([wd1, '\*.csv']);
    t2 = max([fnames2(:).datenum]);
    onedayago = datenum(datetime() - 1);
    %
    if ~isempty(t2) && ...
            (length(fnames)~=length(fnames2)|| t2 > t1) %&& t2 <= onedayago
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
        wd1 = [wd,'\',sname,'\inform_data\Phenotyped'];
        GetaSeg(wd1, sname, MergeConfig);
        %
        wd1 = [wd,'\',sname,'\inform_data'];
        GetnoSeg(wd1, sname, MergeConfig)
        %
    end
end
%
end