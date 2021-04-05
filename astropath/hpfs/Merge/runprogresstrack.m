%% Run Progress Tracker 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 12/19/2018
%% --------------------------------------------------------------
%% Descritption
% For all Clinical Specimens in the folder track progress if the samples
% are ready to be moved forward this function will call the
% necessary functions
%% --------------------------------------------------------------
%% input: 
% The program will ask for 3 directories
%     1. Clincial_Specimen folder
%     2. flatw folder
%     1. transfer path folder
%% output: Batch\sample_specimens.xls
%%% also may create *\Tables and *\flatw files
%% --------------------------------------------------------------
%%
function runprogresstrack(main)
%
tbl = readtable([main, '\Paths.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
%
for i1 = 1:height(tbl)
    %
    % Clinical_Specimen folder
    %
    wd = tbl(i1,'Main_Path');
    wd = table2array(wd);
    wd = wd{1};
    %
    % get the machine name
    %
    machine = tbl(i1,'Machine');
    machine = table2array(machine);
    machine = machine{1};
    %
    % check drive space
    %
    str = strsplit(wd, '\');
    str = ['\\', str{2}, '\', str{3}];
    %
    TB = java.io.File(str).getFreeSpace * (10^-12);
    TB = round(TB, 3);
    %
    tbl(i1,'DriveSpace_TB') = {TB};
    try
        writetable(tbl,[main, '\Paths.csv'])
    catch
    end
    %
    % run progress tracker unless process string is No from paths file
    %
    gostr = tbl(i1,'Process_Merge');
    gostr = table2array(gostr);
    gostr = gostr{1};
    if strcmpi(gostr,'No')
        continue
    end
    %
    progresstrack(main, wd, machine);
    %
    % fill main inForm queue
    %
    try
        pop_main_queue(wd, main)
    catch
    end
end
%
end
%% Progress Tracker for Clinical_Specimens on bki04
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% For all Clinical Specimens in the folder track progress if the samples
%%% are ready to be moved forward this function will call the necessary functions
%% --------------------------------------------------------------
%%
function [] = progresstrack(main, wd, machine)
%
%start by getting all the folder names
%
fn = dir(wd);
fd = fn(3:end);
ii = [fd.isdir];
fd = fd(ii);
samplenames = {fd(:).name};
%
% filter for only Clinical_Specimen folders
%
ii = (contains(samplenames, 'Batch')...
    |contains(samplenames, 'tmp_inform_data')|...
    contains(samplenames, 'reject')|...
    contains(samplenames, 'Control')|...
    strcmp(samplenames, 'Clinical')|...
    strcmp(samplenames, 'upkeep_and_progress')|...
    strcmpi(samplenames, 'Flatfield')|...
    strcmpi(samplenames, 'dbload')|...
    strcmpi(samplenames, 'Crtl')|...
    strcmp(samplenames, 'logfiles'));
samplenames = samplenames(~ii);
%
% make Batch folder
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
    return
end
ii = strcmp(Bt.Opal,'DAPI') | strcmp(Bt.Target,'DNA');
Bt = Bt(~ii,:);
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
Targets =  Bt(:,'Target');
%
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
%
% the name of the summary sheet
%
ssname = [wd,'\upkeep_and_progress\samples_summary.xlsx'];
%
%fill the name vector of the ABs used
%
track = 1;
AB = cell(length(tmpfd)*4,1);
for i1 = 1:length(tmpfd)
    AB{track} = [tmpfd(i1).name,'_Expected_InForm_Files'];
    AB{track+1} = [tmpfd(i1).name,'_Actual_InForm_Files'];
    AB{track+2} = [tmpfd(i1).name,'_Errors_InForm_Files'];
    AB{track+3} = [tmpfd(i1).name,'_InForm_Algorithm'];
    AB{track+4} = [tmpfd(i1).name,'_InForm_date'];
    track = track + 5;
end
%
% create summary spreadsheet
%
tblsz = 25 + length(AB);
ss = array2table(zeros(0,tblsz));
ss.Properties.VariableNames = [{'Machine','Main_Path','Sample','Batch',...
    'Scan','Scan_date','Expected_Im3s','Actual_Im3s','Errors_Im3s','Transfer_Date',...
    'Expected_Flatw_Files','Actual_Flatw_Files','Errors_Flatw_Files','Flatw_Date'},AB',...
    {'All_Expected_InForm_Files','All_Actual_InForm_Files',...
    'All_Errors_InForm_Files','All_InForm_Date','Expected_Merged_Tables',...
    'Actual_Merged_Tables','Errors_Merged_Tables','Merge_Tables_Date',...
    'Actual_QC_Images','QC_Ready_Date','QC_Done_Date'}];
%
%populate the table with all blank rows to start.
%
ss = [ss;[repmat({''},length(samplenames),tblsz)]];
%
%initialize all tracking vectors
%
samplenamesout = cell(1);
BatchID = cell(1);
ScanNum = cell(1);
transferdate = cell(1);
actualim3num = cell(1);
expectim3num = cell(1);
errorim3num = cell(1);
Scandate = cell(1);
actualflatwnum = cell(1);
expectflatwnum = cell(1);
errorflatwnum = cell(1);
flatwdate = cell(1);
actual_infm = cell(1,length(tmpfd));
expect_infm = cell(1,length(tmpfd));
diff_infm = cell(1,length(tmpfd));
infmd = cell(1,length(tmpfd));
algd = cell(1,length(tmpfd));
diff_ifall =  cell(1);
aifall =  cell(1);
exifall =  cell(1);
ifalldate =  cell(1);
actual_merged_tables =  cell(1);
expect_merged_tables =  cell(1);
diff_merged_tables = cell(1);
MergeTblsDate =  cell(1);
QCImagesdate = cell(1);
QCImages = cell(1);
QC_done_date = cell(1);
%
%loop through each sample to track and if data is available to move the
%progress forward; either by running the flat warp or the inform table
%merge/ QC functions
%
for i1 = 1:height(ss)
    %
    % sname will be used as the sample name
    %
    sname = samplenames{i1};
    spath = [wd,'\',sname,'\im3'];
    if ~exist(spath,'dir')
        fprintf(['"',sname,'" is not a valid clinical specimen folder \n']);
        continue
    end
    samplenamesout{i1} = sname;
    %
    % determine the most recent Scan
    %
    [Scanpath, ScanNum{i1}, BatchID{i1}] = getscan(wd, sname);
    %
    % next we look at the annotations files for the number of expected im3
    %
    [expectim3num{i1}] = getAnnotations(Scanpath,['Scan',num2str(ScanNum{i1})],sname);
    %
    % determine when MSI folder was created as Scan date tracker
    %
    MSIpath = [Scanpath, 'MSI*'];
    MSIfolder = dir(MSIpath);
    transferdate{i1} = MSIfolder.date(1:11);
    %
    % get number of im3s
    %
    im3path = [Scanpath,'MSI\*.im3'];
    im3s = dir(im3path);
    actualim3num{i1} = length(im3s);
    errorim3num{i1} = expectim3num{i1} - actualim3num{i1};
    %im3{i1} = [num2str(actualim3num),'of',num2str(expectim3num)];
    %
    % get date of scan start
    %
    [~,idx] = min([im3s(:).datenum]);
    Scandate{i1} = im3s(idx).date(1:11);
    %
    % track flatw files
    %
    [flatwdate{i1}, actualflatwnum{i1}, expectflatwnum{i1}, alg] = getflatws(...
        wd,sname, actualim3num{i1}, main, tmpfd);
    errorflatwnum{i1} = expectflatwnum{i1} - actualim3num{i1};
    algd(i1,:) = alg(:);
    %
    % for inform files
    %
    informpath = [wd,'\',sname,'\inform_data'];
    [actualinform,expectedinform, infma, infmda, trackinform,iffdloc,iffd,...
        expectedTablesnum] = getinformfiles(sname, actualim3num{i1},tmpfd,informpath);
    actual_infm(i1, :)= num2cell(actualinform(:)');
    expect_infm(i1, :)= num2cell(expectedinform(:)');
    diff_infm(i1,:) = num2cell(expectedinform(:) - actualinform(:))';
    infmd(i1,:) = infmda(:)';
    %
    % if trackinform is same as number of tmpfd ABxx then all the inform
    % data is done and it is time to run merge function; track this
    %
    difallfd = '';
    if trackinform == length(tmpfd)
        aifall{i1} = sum(actualinform);
        exifall{i1} = sum(expectedinform);
        diff_ifall{i1} = [exifall{i1} - aifall{i1}];
        iffda = iffd(iffdloc);
        [~,idx] = max([iffda.datenum]);
        difallfd = iffda(idx).date;
        ifalldate{i1} = iffda(idx).date(1:11);
    end
    MergeConfig = [wd,'\Batch\MergeConfig_',BatchID{i1},'.xlsx'];
    %
    % get merge tables info and run MaSS if needed
    %
    [MergeTbls, MergeTblsDate{i1}, Rfd, dRfd] = getmergefiles(...
        sname, informpath, trackinform, tmpfd,...
        difallfd,expectedTablesnum,MaSSpath, MergeConfig);
    if ~isempty(MergeTbls)
        MergeTblsa = strsplit(MergeTbls,'of');
        actual_merged_tables{i1} = str2double(MergeTblsa{1});
        expect_merged_tables{i1} = str2double(MergeTblsa{2});
        diff_merged_tables{i1} = str2double(MergeTblsa{2})...
            - str2double(MergeTblsa{1});
    else
        actual_merged_tables{i1} = 0;
        expect_merged_tables{i1} = 0;
        diff_merged_tables{i1} = 0;
    end
    %
    % get image QC info and run image QC if needed
    %
    [QCImagesdate{i1}, QCImages{i1}]  = ...
        getQCfiles(sname, informpath, Rfd, dRfd, CreateQAQCpath, MergeConfig);
    %
    [QC_done_date{i1}] = getQCstatus(sname, wd, Targets);
    %
    disp(['Completed ',sname, ' Slide Number: ',num2str(i1)]);
end
%
% fill out the progress table
%
ss.Machine(1:length(samplenamesout)) = repmat({machine},length(samplenamesout),1);
ss.Main_Path(1:length(samplenamesout)) = repmat({wd},length(samplenamesout),1);
ss.Sample(1:length(samplenamesout)) = samplenamesout';
ss.Batch(1:length(BatchID)) = BatchID';
ss.Scan(1:length(ScanNum)) = ScanNum';
ss.Scan_date(1:length(Scandate)) = Scandate';
ss.Expected_Im3s(1:length(expectim3num)) = expectim3num';
ss.Actual_Im3s(1:length(actualim3num)) = actualim3num';
ss.Errors_Im3s(1:length(expectim3num)) = errorim3num';
ss.Transfer_Date(1:length(transferdate)) = transferdate';
ss.Expected_Flatw_Files(1:length(expectflatwnum)) = expectflatwnum';
ss.Actual_Flatw_Files(1:length(actualflatwnum)) = actualflatwnum';
ss.Errors_Flatw_Files(1:length(expectflatwnum)) = errorflatwnum;
ss.Flatw_Date(1:length(flatwdate)) = flatwdate';
tt = 1;
for i1 = 1:length(tmpfd)
    dd = length(actual_infm(:,i1));
    ss.(AB{tt})(1:dd) = expect_infm(:,i1);
    id = AB{tt+1};
    ss.(id)(1:dd) = actual_infm(:,i1);
    id = AB{tt+2};
    ss.(id)(1:dd) = diff_infm(:,i1);
    id = AB{tt+3};
    ss.(id)(1:dd) = algd(:,i1);
    id = AB{tt+4};
    ss.(id)(1:dd) = infmd(:,i1);
    tt = tt + 5;
end
ss.All_Expected_InForm_Files(1:length(exifall)) = exifall';
ss.All_Actual_InForm_Files(1:length(aifall)) = aifall';
ss.All_Errors_InForm_Files(1:length(diff_ifall)) = diff_ifall';
ss.All_InForm_Date(1:length(ifalldate)) = ifalldate';
%
ss.Expected_Merged_Tables(1:length(expect_merged_tables)) = expect_merged_tables';
ss.Actual_Merged_Tables(1:length(actual_merged_tables)) = actual_merged_tables';
ss.Errors_Merged_Tables(1:length(diff_merged_tables)) = diff_merged_tables';
ss.Merge_Tables_Date(1:length(MergeTblsDate)) = MergeTblsDate';
%
ss.Actual_QC_Images(1:length(QCImages)) = QCImages';
ss.QC_Ready_Date(1:length(QCImagesdate)) = QCImagesdate';
ss.QC_Done_Date(1:length(QC_done_date)) = QC_done_date';
%
%save the progress table in Batch folder as sample summary
%
try
    writetable(ss,ssname)
catch
    pause(10*60)
    try
        writetable(ss,ssname)
    catch
    end
end
    %
end
%% pop_main,queue
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% populate the main inForm_queue if the files have not yet been processed,
%%% queue location is in the 'main' folder
%% --------------------------------------------------------------------
%%
function[] = pop_main_queue(wd, main)
    %
    % if main_inform_queue does not exist create it
    %
    str = 'Path,Specimen,Antibody,Algorithm,Processing \r\n';
    %
    if ~exist([main,'\inForm_queue.csv'], 'file')
        f = fopen([main,'\inForm_queue.csv'], 'w' );  
           fprintf(f,str);
        fclose(f);
    end
    %
    % if clinical specimen inForm queue does not exist create it
    %
    if ~exist([wd,'\upkeep_and_progress\inForm_queue.csv'], 'file')
            f = fopen([wd,'\upkeep_and_progress\inForm_queue.csv'], 'w' );
            fprintf(f,str);
            fclose(f);
    end
    %
    str = 'Path,Specimen,Antibody,Algorithm,Processing ';
    %
    % open the main inForm queue
    %
    fileID = fopen([main,'\inForm_queue.csv']);
    queuef = textscan(fileID,'%s','HeaderLines',...
        1,'EndofLine','\r', 'whitespace','\t\n');
    fclose(fileID);
    queuef_main = queuef{1};
    %
    % trim all the white space and if the string contains two 'Processing'
    % strings assume the second string ran properly 
    %
    del_rows = [];
    %
    for i1 = 1:length(queuef_main)
        x = queuef_main{i1};
        x = strsplit(x,',');
        if isempty(x{1}) || length(x{1}) > 2000
            del_rows = [del_rows,i1];
            continue
        end
        x = cellfun(@(x)strtrim(x),x,'Uni',0);
        x = strjoin(x,', ');
        if count(x,'Processing') > 1
            str1 = extractBetween(x,'Processing','Processing');
            x = replace(x,strcat('Processing',str1),'');
        end
        queuef_main{i1} = x;
    end
    %
    if ~isempty(del_rows)
        queuef_main(del_rows) = [];
    end
    %
    % open the clinical specimen inForm queue
    %
    fileID = fopen([wd,'\upkeep_and_progress\inForm_queue.csv']);
    queuef = textscan(fileID,'%s','HeaderLines',...
        1,'EndofLine','\r', 'whitespace','\t\n');
    fclose(fileID);
    queuef_CS = queuef{1};
    %
    % trim all the white space and if the string contains two 'Processing'
    % strings assume the second string ran properly 
    %
    del_rows = [];
    queuef_CS_check = cell(length(queuef_CS),1);
    %
    for i1 = 1:length(queuef_CS)
        x = queuef_CS{i1};
        x = strsplit(x,',');
         if isempty(x{1}) || length(x{1}) > 2000
            del_rows = [del_rows,i1];
            continue
        end
        x = cellfun(@(x)strtrim(x),x,'Uni',0);
        queuef_CS_check{i1} = strjoin(x(1:3),', ');
        x = strjoin(x,', ');
        if count(x,'Processing') > 1
            str1 = extractBetween(x,'Processing','Processing');
            x = replace(x,strcat('Processing',str1),'');
        end
        queuef_CS{i1} = x;
    end
    %
    % remove blank rows
    %
    if ~isempty(del_rows)
        queuef_CS(del_rows) = [];
        queuef_CS_check(del_rows) = [];
    end
    %
    % get the unique lines from CS queue
    %
    D = unique(queuef_CS_check(:));
    %
    % update the inform queues for lines that have been added more than once 
    %
    for i1 = 1:length(D)
        %
        % check for each unique line in both queues
        %
        current_line = D(i1);
        ii = strcmp(current_line, queuef_CS_check);
        CS_line_numbers = find(ii);
        CS_line_counts = numel(CS_line_numbers);
        %
        ii = contains(queuef_main,[current_line{1},',']);
        M_line_numbers = find(ii);
        M_line_counts = numel(M_line_numbers);
        %
        % for each line in CS queue check if the number of lines is less
        % than the number in main queue. If it is then assume the rows are
        % in matching order. Check if the current line in CSn or in main 
        % queue is longer; if it is longer in the CS queue replace
        % in the main queue otherwise use what is in the main queue for
        % both. If the number of matching lines found in the CS queue 
        % exceeds the length of lines found in the main queue add the end 
        % of the main queue.
        %
        for i2 = 1:CS_line_counts
            CSn = queuef_CS(CS_line_numbers(i2));
            len_CSn = length(CSn{:});
            %
            if i2 <= M_line_counts
                Mn = queuef_main(M_line_numbers(i2));
                len_Mn = length(Mn{:});
                %
                if len_CSn > len_Mn
                    queuef_main(M_line_numbers(i2)) = CSn;
                else
                    queuef_CS(CS_line_numbers(i2)) = Mn;
                end
            else
                queuef_main(end+1) = CSn;
            end
        end
        %
        if M_line_counts > CS_line_counts
            for i2 = CS_line_counts + 1:M_line_counts
                Mn = queuef_main(M_line_numbers(i2));
                queuef_CS(end+1) = Mn;
            end
        end
    end
    %
    [nrow, ncol] = size(queuef_main);
    if ncol > nrow
        queuef_main = queuef_main';
    end
    queuef_main = replace(queuef_main,'\','\\');
    queuef_main = [{str};queuef_main];
    %
    queuef_CS = replace(queuef_CS,'\','\\');
    queuef_CS_out = [{str};queuef_CS];
    %
    % remove the old file and write a new one
    %
    try
        %
        f = fopen([wd,'\upkeep_and_progress\inForm_queue.csv'], 'w' );
        cellfun(@(x) fprintf(f,[x,' \r\n']),queuef_CS_out);
        fclose(f);
        %
    catch
    end
    % 
    try
        %
        qf = [wd,'\upkeep_and_progress\Main_inForm_queue.csv'];
        mf = [main,'\inForm_queue.csv'];
        f = fopen(qf, 'w' );
        cellfun(@(x) fprintf(f,[x,' \r\n']),queuef_main);
        fclose(f);
        %
        copyfile(qf,mf);
    catch
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
         try
             s = temp.getAttribute('subtype');
         catch
             continue;
         end
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
%%% read the annotation for update inForm verions
%% --------------------------------------------------------------
%%
function F = ROIAnnotationRead(ann)
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
%% getflatws
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% for a give specimen in a directory; check if the status of the
%%% flatwarping of the im3 images
%% --------------------------------------------------------------
%%
function[flatwdate,actualflatwnum, expectflatwnum, alg] = ...
    getflatws(wd,sname, actualim3num, main,tmpfd)
        flatwdate = [];
        actualflatwnum = 0;
        expectflatwnum = 0;
        alg = repmat({''},1,numel(tmpfd));
        %
        % flatw im3 location path
        %
        flatwpath = [wd,'\',sname,'\im3\flatw'];
        %
        % find the number of im3 files created and their date
        %
        if exist(flatwpath,'dir')
            flatw = dir([flatwpath,'\*.im3']);
            %
            % check if flatw ran for all images 
            %
            if length(flatw) ~= actualim3num
                %
                % if number of *.im3's do not equal number of flatw *.im3's
                % then there was an error in flat code or it is not finished
                % delete flatw tracking and move on 
                %
                flatw = [];
                flatwdate = 'NA';
                %}
            else
                %
                % if flatw code ran correctly use most recent date
                %
                [~,idx] = max([flatw(:).datenum]);
                flatwdate = flatw(idx).date(1:11);
                %
                % populate that Specimen to the inForm_queue if its not
                % already there
                %
                alg = pop_queue(wd, sname, main, tmpfd);
            end
            actualflatwnum = length(flatw);
            expectflatwnum = actualim3num;
        end
end
%% pop_queue
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% populate the inForm_queue if the files have not yet been processed,
%%% queue location is in the 'main' folder
%% --------------------------------------------------------------------
%%
function alg = pop_queue(wd, sname, main, tmpfd)
    %
    % first check if any inform output already exists
    %
    try
        alg = findalg(wd,sname, tmpfd);
    catch
        alg = repmat({''},1,numel(tmpfd));
    end
    ii = zeros(length(tmpfd),1);
    for i2 = 1:length(tmpfd)
        AB = tmpfd(i2).name;
        wd1 = [wd,'\',sname,'\inform_data\Phenotyped\',AB];
        ii(i2) = exist(wd1, 'dir');
    end
    ii2 = find(~ii);
    if sum(ii2) == 0
        return
    end
    %
    % if inform_queue does not exist create it
    %
    if ~exist([wd,'\upkeep_and_progress\inForm_queue.csv'], 'file')
        f = fopen([wd,'\upkeep_and_progress\inForm_queue.csv'], 'w' );  
         str = 'Path,Specimen,Antibody,Algorithm,Processing \r\n';
           fprintf(f,str);
        fclose(f);
    end
    %
    % open the inForm queue
    %
    fileID = fopen([wd,'\upkeep_and_progress\inForm_queue.csv']);
    queuef = textscan(fileID,'%s','HeaderLines',...
        1,'EndofLine','\r', 'whitespace','\t\n');
    fclose(fileID);
    queuef = queuef{1};
    queuef = cellfun(@(x)strrep(x,' ' ,''),queuef,'Uni',0);
    
    %
    % add those files to the queue if they are not already there
    %
    NewABs = tmpfd(ii2);
    %
    for i2 = 1:length(NewABs)
        if ~sum(contains(queuef,[sname,',',NewABs(i2).name,',']),1)
            i3 = replace(wd,'\','\\');
            fileID = fopen([wd,'\upkeep_and_progress\inForm_queue.csv'], 'a');
            AB = NewABs(i2).name;
            str = [i3,',',sname,',',AB,', , \r\n'];
            try
                fprintf(fileID,str);
                fclose(fileID);
            catch
            end
        end
    end
    alg = findalg(wd,sname,tmpfd);
end
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
    = getinformfiles(sname, actualim3num, tmpfd, informpath)
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
    errs = [];
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
        %
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
        %
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
                if ~isempty(Bf)
                    fileID = fopen([Bf.folder,'\',Bf.name]);
                    Batch = textscan(fileID,'%s','HeaderLines',...
                        2,'EndofLine','\r', 'whitespace','\t\n');
                    fclose(fileID);
                    Batch = Batch{1};
                    %
                    ii = contains(Batch,sname);
                    Batch = Batch (ii);
                    Batch = extractBefore(Batch, ']');
                    %
                    ii = unique(extractAfter(Batch, '_['));
                    InformErrors = length(ii);
                    %
                    errs = [errs;ii];
                    expectedinform(i2) = actualim3num - InformErrors;
                else
                    expectedinform(i2) = actualim3num;
                end
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
        end
    end
    if ~isempty(gcp('nocreate'))
        poolobj = gcp('nocreate');
        delete(poolobj);
    end
    expectedTablesnum = actualim3num;
    if ~isempty(errs)
        errs = unique(errs);
        expectedTablesnum = expectedTablesnum - length(errs);
    end
    
end
%% Transfer files between directories fast
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 12/13/2018
%% --------------------------------------------------------------------
%% Description
%%% transfer all files from sorce folder to destination folder using a
%%%parfor loop
%% input
%%% sor: the sorce directory
%%% des: the destination directory
%% --------------------------------------------------------------------
%%
function [C] = transferfls(sor,des)
cfiles = dir(sor);
ii = strcmp({cfiles(:).name}, '.')|strcmp({cfiles(:).name},'..');
cfiles = cfiles(~ii);
C = ~isempty(cfiles);
if C
    if ~exist(des,'dir')
        mkdir(des)
    end
    parfor i3 = 1:length(cfiles)
        sor = [cfiles(i3).folder,...
            '\', cfiles(i3).name];
        try
            movefile(sor, des)
        catch
        end
    end
end
end
%% getmergefiles
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% runs the merge code for a specimen if it has not been run before or if
%%% there is newer inform data in one of the ABx folders that the results,
%%% it will also track the number of merged files and merge dates
%% --------------------------------------------------------------------
%%
function [MergeTbls, MergeTblsDate, Rfd, dRfd] = ...
    getmergefiles(sname, informpath,trackinform, tmpfd,...
    difallfd,expectedTablesnum,MaSSpath,MergeConfig)     
    %
    MergeTblsDate = [];
    MergeTbls = [];
    dRfd = [];
    %
    % if Results folder does not exist but there are enough inform files to
    % generate output then run merge function
    %
    Rfd = [informpath,'\Phenotyped\Results\Tables'];
    mergeroot = informpath;
    if ~exist(Rfd,'dir') && trackinform == length(tmpfd)
        command = ['CALL "', MaSSpath, '" "',mergeroot,'" "', sname,'" "',...
            MergeConfig,'"'];
        system(command);
    end
    lf = fullfile(Rfd,'MaSSLog.txt');
    if exist(lf, 'file')
       fid = fopen(lf);
       if fid >= 0
           erl = textscan(fid,'%s','HeaderLines',...
               2,'EndofLine','\r', 'whitespace','\t\n');
           fclose(fid);
           %
           erl = erl{1};
           ii = contains(erl,'Error');
           if ~isempty(find(ii,1))
               delete([Rfd,'\*'])
           end
       end
    end
    %
    % check if folder exists
    %
    if exist(Rfd,'dir')
        % 
        Tablespath = [informpath,'\Phenotyped\Results\Tables\*_table.csv'];
        Tblsdate = dir(Tablespath);
        %
        Tablesnum = length(Tblsdate);
        %
        if ~isempty(Tblsdate)
            [~,idx] = max([Tblsdate.datenum]);
            Tblsdate = Tblsdate(idx);
            %
            dRfd = Tblsdate.date;
        end
        %
        % if results folder was created after most recent inform folder
        % rerun merge functions and create new results
        %
        if isempty(Tblsdate) || (datetime(dRfd) < datetime(difallfd)) ...
                || Tablesnum ~= expectedTablesnum
            command = ['CALL "', MaSSpath, '" "',mergeroot,'" "', sname,'" "',...
            MergeConfig,'"'];
            system(command);
            Tblsdate = dir(Tablespath);
            %
            if ~isempty(Tblsdate)
                [~,idx] = max([Tblsdate.datenum]);
                Tblsdate = Tblsdate(idx);
                %
                dRfd = Tblsdate.date;
            end
        end
        %
        % set up tracking functions
        %
        if ~isempty(Tblsdate)
            MergeTblsDate = Tblsdate.date(1:11);
            %
            Tablespath = [informpath,'\Phenotyped\Results\Tables\*_table.csv'];
            Tables = dir(Tablespath);
            Tablesnum = length(Tables);
            %
            MergeTbls = [num2str(Tablesnum),'of',num2str(expectedTablesnum)];
        end
    end
end
%% getQCfiles
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% runs the QA_QC code for a specimen if it has not been run before or if
%%% there is newer results tables,
%%% it will also track the number of QA_QC files and QA_QC ready dates
%% --------------------------------------------------------------------
%%
function [QCImagesdate, QCImages]  = getQCfiles(sname, informpath,...
    Rfd, dRfd, CreateQAQCpath, MergeConfig)    
    QCImagesdate = [];
    QCImages = [];
    mergeroot = informpath;
    %
    % if QC folder does not exist but the Results folder does then run
    % Image function
    %
    QCfd = [informpath,'\Phenotyped\Results\QA_QC'];
    QCfl = dir([informpath,'\Phenotyped\Results\tmp_ForFiguresTables\*.mat']);
    %
    if exist(Rfd,'dir')
        Rfd = dir([Rfd,'\*table.csv']);
    else
        Rfd = [];
    end
    %
    if ~isempty(Rfd) && ~exist(QCfd,'dir') && ~isempty(QCfl)
        %
        % call image loop
        %
        command = ['CALL "', CreateQAQCpath, '" "',mergeroot,'" "', sname,'" "',...
            MergeConfig,'"'];
        system(command);
        %
    elseif ~isempty(Rfd) && ~exist(QCfd,'dir') && isempty(QCfl)
         QCImages = 0;
    end
    if exist(QCfd,'dir')
        %
        % create QC path and get images 
        %
        QCfd = [QCfd,'\Phenotype\All_Markers\*composite_image.tif'];
        QCfiles = dir(QCfd);
        %
        QCfl = dir([informpath,'\Phenotyped\Results\tmp_ForFiguresTables']);
        QCfl = QCfl(3:end);
        %
        % if QC output is empty and there are tmp_ForFigureTables then try
        % to make QC output again
        %
        if isempty(QCfiles) && ~isempty(QCfl)
            command = ['CALL "', CreateQAQCpath, '" "',mergeroot,'" "', sname,'" "',...
            MergeConfig,'"'];
            system(command);
            %
            QCfiles = dir(QCfd);
        end
        %
        if ~isempty(QCfiles) 
            %
            % get date of most recent file
            %
            [~,idx] = max([QCfiles(:).datenum]);
            dQCfd = QCfiles(idx).date;
            %
            % check if Results are newer than QC folder, if they are then
            % create a new QC output
            %
            QCfl = dir([informpath,'\Phenotyped\Results\tmp_ForFiguresTables']);
            QCfl = QCfl(3:end);
            %
            if datetime(dRfd) > datetime(dQCfd)
                if ~isempty(QCfl)
                    command = ['CALL "', CreateQAQCpath, '" "',mergeroot,...
                        '" "', sname,'" "',MergeConfig,'"'];
                    system(command);
                else 
                    rmdir(extractBefore(QCfd,'\QA_QC\'),'s');
                end
                %
                QCfiles = dir(QCfd);
                %
                % get a new date of most recent file created
                %
                [~,idx] = max([QCfiles(:).datenum]);
            end
            %
            % get number of files that QC was generated on
            %
            if ~isempty(QCfiles)
                QCImagesdate = QCfiles(idx).date(1:11);
                QCImages = length(QCfiles);
            end
        end
        %
    end
end
%% QC_done_date
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% checks the status of the inForm_QC.csv file, if there is a date in the
%%% last column then it output at QC_done_date
%% --------------------------------------------------------------------
%%
function QC_done_date = getQCstatus(sname, wd, Targets)
%
% get QC done date from tbl if it exists
% 
QC_done_date = '';
%
wd1 = [wd, '\upkeep_and_progress\inform_QC.csv'];
%
if ~exist(wd1, 'file')
     tbl = array2table(zeros(0,length(Targets) + 4));
     tbl.Properties.VariableNames = [{'Sample'},Targets',...
         {'QC_done_date', 'Initials','Comments'}];
     tbl = [tbl;[sname,repmat({''},1,length(Targets) + 3)]];
     try
        writetable(tbl, wd1)
     catch
         return
     end
else
    tbl = readtable(wd1,'Delimiter',',',...
        'TreatAsEmpty',{' ','#N/A'}, ...
        'Format', (repmat('%s', 1, length(Targets) + 4)));
    %
    try
        tbl1 = tbl(strcmp(tbl.Sample, sname),'QC_done_date');
    catch
        tbl1 = [];
    end
        %
    if ~isempty(tbl1)
        QC_done_date = table2array(tbl1);
    else
        try
            tbl = [tbl; [sname,repmat({''},1,length(Targets) + 3)]];
            writetable(tbl, wd1)
        catch
            try
                tbl.QC_done_date = repmat({''},height(tbl), 1);
                tbl = [tbl; [sname,repmat({''},1,length(Targets) + 1)]];
                writetable(tbl, wd1)
            catch
                return
            end
        end
    end
end
%
end
%% findlag
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% find the most recent algorithm used to analyze an antibody in inform
%% --------------------------------------------------------------------
%%
function alg = findalg(wd,sname,tmpfd)
    %
    % for each antibody find the algorithm related to the most recent
    % encoded line
    %
    alg = repmat({''},1,numel(tmpfd));
    fileID = fopen([wd,'\upkeep_and_progress\inForm_queue.csv']);
    queuef = textscan(fileID,'%s','HeaderLines',...
        2,'EndofLine','\r', 'whitespace','\t\n');
    fclose(fileID);
    queuef = queuef{1};
    %
    for i2 = 1:length(tmpfd)
        AB = tmpfd(i2).name;
        str = [wd,', ',sname,', ',AB,','];
        line1 = queuef(contains(queuef,str));
        %
        if ~isempty(line1)
            line1 = line1{end};
            line1 = extractBetween(line1,[AB,','],',');
            alg(i2) = line1;
        end
    end
end