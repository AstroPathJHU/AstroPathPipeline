%% intialize folders
%
function [e_val, tmpfd, Targets] =  intialize_folders(wd)
% make Batch folder
%
e_val = 0;
%
Batchfolder = [wd,'\Batch'];
if ~exist(Batchfolder,'dir')
    mkdir(Batchfolder)
end
%
% populate AB calls from one of the Batch tables
%
BTnms = dir([wd,'\Batch\MergeConfig*.xlsx']);
try
    warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')
    Bt = readtable([wd,'\Batch\',BTnms(1).name]);
catch
    e_val = 1;
    return
end
%
% DAPI does not get phenotyped
%
ii = strcmp(Bt.Opal,'DAPI') | strcmp(Bt.Target,'DNA');
Bt = Bt(~ii,:);
%
% multiple segmentations need multiple folders
%
SS = Bt.NumberofSegmentations;
if iscell(SS)
    SS = cell2mat(SS);
    SS = str2double(SS);
end
SS = SS > 1;
idx = find(SS);
%
if ~isempty(idx)
    for i2 = 1:length(idx)
        crow = Bt(idx(i2),:);
        nsegs = crow.NumberofSegmentations;
        if iscell(nsegs)
            nsegs = cell2mat(nsegs);
            nsegs = str2double(nsegs);
        end
        for i3 = 2:nsegs
            crow_out = crow;
            crow_out.Target = {[crow_out.Target{1},'_',num2str(i3)]};
            Bt(end + 1,:) = crow_out;
        end
    end
end
%
% tumor name should be 'Tumor' not antibody for usability
%
Targets =  Bt(:,'Target');
ii = strcmp(Bt.ImageQA,'Tumor');
Targets(ii,:) = {'Tumor'};
Targets = table2array(Targets);
%
% create the tmp_inform_data subdirectory if it does not exist and
% intialize the inform output folders
%
tmppath = [wd,'\tmp_inform_data'];
if ~exist(tmppath,'dir')
    mkdir(tmppath)
end
for i1 = 1:length(Targets)
    cdir = [tmppath,'\',Targets{i1}];
    if ~exist(cdir,'dir')
        mkdir(cdir)
        mkdir([cdir,'\1']) 
    end
end
%
% intialize project development folder
%
cdir = [tmppath,'\Project_Development'];
if ~exist(cdir,'dir')
    mkdir(cdir)
end
%
% get the Info path for all record keeping information
%
infopath = [wd,'\upkeep_and_progress'];
if ~exist(infopath,'dir')
    mkdir(infopath)
end
%
% get the ABs used from the tmp_inform_data subfolder names
%
tmpfd = dir(tmppath);
tmpfd = tmpfd(3:end);
%
%take only antibody directory names
%
ii = [tmpfd.isdir];
tmpfd = tmpfd(ii);
ii = ismember({tmpfd(:).name},Targets);
tmpfd = tmpfd(ii);

end