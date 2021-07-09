%% processflatw
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% for a give specimen in a directory; check if the status of the
%%% flatwarping of the im3 images
%% --------------------------------------------------------------
%%
function process_flatw_queue(main)
%
% read Paths table
%
tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
tbl2 = readtable([main, '\AstropathCohortsProgress.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
tbl3 = readtable([main, '\AstropathConfig.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
%
% check if process flatw queue table exists, if it doesn't create is then
% read it in
%
colheads = CreateflatwQueue(main);
pqt = readtable([main,'\across_project_queues\process_flatw_queue.csv'],'Delimiter', ',',...
        'ReadVariableNames',1,'format',repmat('%s ',1,16));
%
for i1 = 1:height(tbl)
    tic
    %
    [wd, fwpath, machinename, process, samplenames] = getmyVars(tbl, tbl2, tbl3, i1);
    if strcmpi(process,'No')
        continue
    end
    %
    % cycle through each sample track all relavant information 
    %
    for i2 = 1:length(samplenames)
        %
        sname = samplenames{i2};
        if isempty(dir([wd,'\',sname,'\im3\Scan*'])) ||...
                isempty(dir([wd,'\',sname,'\im3\**\BatchID.txt']))
            fprintf(['"',sname,...
                '" is not a valid clinical specimen folder \n']);
            continue
        end
        %
        [ScanNum,BatchID,expectim3num,actualim3num, Scandate,...
            Scanpath, transferdate] = extractRecords(wd, sname);
        %
        % first check if mbin exists then add to pop queue
        %
        [pqt] = pop_processing_queue(pqt, wd, sname, fwpath,...
            machinename, ScanNum, BatchID, expectim3num, actualim3num,...
            Scandate, Scanpath, transferdate, colheads);
        %
    end
    %
    disp(['Processing flatw queue updated for specimens in ',wd]);
    fprintf('           ');
    toc
    %
    try
        writetable(pqt, [main,'\across_project_queues\process_flatw_queue.csv']);
    catch
    end
    %
end
end
%%
function colheads = CreateflatwQueue(main)
%%
% if the process flatw queue does not exist create it
% 
%%
colheads = {'Machine','Main_Path','Sample','Batch',...
    'Scan','Scan_date','expected_im3','actual_im3','transfer_date',...
    'Flatw_Path','flatfield_binfile_date','Processing_flatw_Sent',...
    'Processing_flatw_Start','Processing_flatw_Finished',...
    'expected_flatw_files','actual_flatw_files'};
%
% if processing_queue does not exist create it
%
if ~exist([main,'\across_project_queues'],'dir')
    mkdir([main,'\across_project_queues'])
end

if ~exist([main,'\across_project_queues\process_flatw_queue.csv'], 'file')
    disp(['Processing_queue.csv not found on ', main,'\across_project_queues']);
    disp('Creating ...');
    pqt = cell2table(cell(0,16), 'VariableNames', colheads);
    writetable(pqt, [main,'\across_project_queues\process_flatw_queue.csv']);
end
%
end
%%
function [wd, fwpath, machinename, process, sn] = getmyVars(tbl, tbl2,  tbl3, i1)
%%
% get the variables for the current cohort, i1
%
%%
%
% Clinical_Specimen folder
%
tb = tbl(i1,:);
wd = ['\\', tb.Dpath{1},'\', tb.Dname{1}];
%
% flatw folder
%
fwpath = ['\\', tb.FWpath{1}];
%
project = tb.Project;
tb2 = tbl2(tbl2.Project == project, :);
machinename = tb2.Machine{1};
%
tb3 = tbl3(tbl3.Project == project, :);
process = tb3.Process_Merge{1};
%
% get specimen names for the CS
%
sp = dir(wd);
sp = sp(3:end);
ii = [sp.isdir];
sp = sp(ii);
sn = {sp(:).name};
ii = (contains(sn, 'Batch')|...
    strcmp(sn, 'Clinical')|...
    contains(sn, 'Control')|...
    strcmpi(sn, 'Ctrl')|...
    strcmpi(sn, 'dbload')|...
    strcmpi(sn, 'Flatfield')|...
    strcmpi(sn, 'logfiles')|...
    strcmpi(sn, 'reject')|...
    contains(sn, 'tmp_inform_data')|...
    strcmp(sn, 'Upkeep and Progress')|...
    strcmpi(sn, 'upkeep_and_progress'));
sn(ii) = [];
%
end
%%
function [ScanNum,BatchID,expectim3num,actualim3num, Scandate,...
    Scanpath, transferdate] = extractRecords(wd, sname)
%%
% get relevant directory information 
%
%%
%
% determine the most recent Scan
%
[Scanpath, ScanNum, BatchID] = getscan(wd, sname);
%
% determine when MSI folder was created as Scan date tracker
%
MSIpath = [Scanpath, 'MSI*'];
MSIfolder = dir(MSIpath);
transferdate = MSIfolder.date(1:11);
%
% get expected number of im3s from the annotations
%
[expectim3num] = getAnnotations(Scanpath,['Scan',num2str(ScanNum)],sname);
%
% get the number of im3s in a folder
%
im3path = [Scanpath,'MSI\*.im3'];
im3s = dir(im3path);
actualim3num = length(im3s);
%
% get date of scan start
%
[~,idx] = min([im3s(:).datenum]);
Scandate = im3s(idx).date(1:11);
%
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
if isempty(Scan)
    Scanpath = [];
    ScanNum = [];
    BatchID = [];
    return
end
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
%%----------------------------------------------------
%% read XML file with grid annotations
%% Daphne Schlesinger, 2017
%% edited by Alex Szalay, 2018-0725
%% edited by Benjamin Green, 2018-1212
%%----------------------------------------------------
%% take the path for an xmlfile corresponding to a qptiff.
%% and produce the number of expected im3's from the annoatations
%%------------------------------------------------------
function [expectim3num] = getAnnotations(root,scan,smp)
filepath = [root,smp,...
    '_', scan '_annotations.xml'];
%
XML = xmlread(filepath);
annList = XML.getFirstChild;
ann = annList.item(5);
% for phenochart update
temp = ann.item(1);
%for i2 = 1:temp.getLength
s = temp.getAttribute('subtype');
if s.equals("ROIAnnotation")
   F = ROIAnnotationRead(ann);
else
%
B = cell(1);
track = 1;
%
% get rectangular annotations
%
for i1 = 1:2: ann.getLength - 1
    temp = ann.item(i1);
    s = temp.getAttribute('subtype');
    
    % try
    %     s = temp.getAttribute('subtype');
    % catch
    %     continue;
    % end
    if  s.equals("RectangleAnnotation")
        B{track} = temp;
        track = track + 1;
    else
    end
end
%
F = cell(1);
track2 = 1;
for i2 = 1 : length(B)
    %
    node = B{i2};
    history = node.item(7);
    histlastRef = history.getLength-2;
    histRef = history.item(histlastRef);
    %
    f =  histRef.item(3).getTextContent;
    t =  histRef.item(7).getTextContent;
    if strcmp(t, 'Acquired')
        f = char(f);
        F{track2} = f(5:end);
        track2 = track2 + 1;
    end
    %
end
%
end
expectim3 = cellfun(@(x)erase(x,'_M2'),F,'Uni',0);
expectim3num = length(unique(expectim3));
end
%% ROIAnnotationRead
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/08/2020
%% --------------------------------------------------------------
%% Description:
%%% read the annotation for update inForm verision
%% --------------------------------------------------------------
%%
function F = ROIAnnotationRead(ann)
%
F = cell(1);
track2 = 1;
%
for i4 = 1:2:ann.getLength
    try
        s = ann.item(i4).getAttribute('subtype');
    catch
        continue
    end
    
    if s.equals("ROIAnnotation") || s.equals('TMASectorAnnotation')
        for i1 = 1:2: ann.item(i4).item(13).getLength - 1
            temp = ann.item(i4).item(13).item(i1);
            s = temp.getAttribute('subtype');
            if  s.equals("RectangleAnnotation")
                node = temp;
                %
                history = node.item(7);
                histlastRef = history.getLength-2;
                histRef = history.item(histlastRef);
                %
                f =  histRef.item(3).getTextContent;
                t =  histRef.item(7).getTextContent;
                if strcmp(t, 'Acquired')
                    f = char(f);
                    F{track2} = f(5:end);
                    track2 = track2 + 1;
                end
            end
        end
        %
    end
end
end
%%
%% pop_processing_queue
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% populate the processing flatw queue if the files have not yet been processed,
%%% queue location is in the 'main' folder
%% --------------------------------------------------------------------
%%
function[pqt] = pop_processing_queue(pqt, wd, sname, fwpath,machinename,...
    ScanNum,BatchID,expectim3num,actualim3num, Scandate, Scanpath,...
    transferdate, colheads)
%
% add those files to the queue if they are not already there update if it
% is
%
str = cell2table({machinename,wd,sname,BatchID,ScanNum,...
    Scandate,expectim3num,actualim3num,transferdate,fwpath,...
    '','','','','',''}, 'VariableNames',colheads);
%
ii = strcmp(pqt.Sample,sname);
if isempty(pqt) || ~sum(ii)
    pqt = [pqt;str];
else
    pqt.actual_im3(ii) = {str.actual_im3};
end
%
mbin = [wd, '\flatfield\flatfield_BatchID_',BatchID,'.bin'];
%
ii = strcmp(pqt.Sample,sname);
pqt.flatfield_binfile_date{ii} = {''};
pqt.Processing_flatw_Finished{ii} = {''};
pqt.actual_flatw_files{ii} = {''};
pqt.expected_flatw_files{ii} = {''};
%
if exist(mbin,'file')
    %
    mbin = dir(mbin);
    mbindf = mbin.date;
    mbind = mbindf(1:11);
    %
    pqt.flatfield_binfile_date{ii} = {mbind};
    %
    flatwpath = [extractBefore(Scanpath,'Scan'),'flatw'];
    if exist(flatwpath,'dir')
        flatwf = [flatwpath,'\*.im3'];
        %
        flatw = dir(flatwf);
        flatwnum = length(flatw);
        [~,idx] = min([flatw(:).datenum]);
        flatwdf = flatw(idx).date;
        %
        if actualim3num  == flatwnum
            dd = extractBefore(Scanpath,'\Scan');
            dd = dir(dd);
            idx = strcmp({dd(:).name},'flatw');
            %
            flatwdf = dd(idx).date(1:11);
            pqt.Processing_flatw_Finished{ii} = {flatwdf};
            pqt.actual_flatw_files{ii} = {num2str(flatwnum)};
            pqt.expected_flatw_files{ii} = {num2str(actualim3num)};
            %
        end
    end    
end
end