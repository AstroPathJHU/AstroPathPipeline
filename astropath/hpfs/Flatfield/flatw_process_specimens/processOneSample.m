%%
%% processOneSample
%% -----------------------------------------------------
%% Created by: Benjamin Green - 05/10/2019 -- Johns Hopkins University
%% -----------------------------------------------------
%% Description
% processing code for drives on bki05; this code allows processing of an
% individual specimen on a bki drive
%% -----------------------------------------------------
%% Usage
% dd = 'C:\Users\bgreen42\Desktop\ProcessingUpdate'
%
function processOneSample(dd)
%
% get the process flatw queue
%
[pqt,~,p2] = getpqt(dd, 1);
if p2 == 0
    fprintf('ERROR IN PQT. EXITING FUNCTION...\n')
    return
end
%
% find the idicies of the process flatw queue that have been sent to the
% drive with a flatfield bin file created but do not have a finished data
%
idx = find(strcmp(pqt.Processing_flatw_Finished,'') &...
    ~strcmp(pqt.Sample,'') &...
    ~strcmp(pqt.flatfield_binfile_date,''), 1);
%
if idx
    %
    % intialize variables from the table
    %
    sid = pqt.Sample{idx};
    wd = pqt.Main_Path{idx};
    machinename = pqt.Machine{idx};
    [Scanpath, ScanNum, BatchID] = getscan(wd, sid);
    %
    % copy files onto local drive for flat fielding
    %
    SETUPdir(wd, dd, sid, ScanNum, Scanpath, BatchID);
    %
    % main flatfielding loop
    %
    [p, flatwfiles,actualim3num]...
        = startOneflat(machinename, dd, sid, idx, pqt,ScanNum);
    %
    % grab a new doc since the old may not have new data in it
    %
    [pqt,pqf, p2] = getpqt(dd, 2);
    if p2 == 0
        fprintf('ERROR IN PQT. EXITING FUNCTION...\n')
        return
    end
    %
    % if there was an error clear start date and exit otherwise try to
    % transfer data back to the main paths
    %
    if p == 0
        pqt.Processing_flatw_Start(idx) = {''};
        p2 = tryCATCHwritetable(pqt, pqf);
        if p2 == 0
            fprintf('ERROR IN PQT. EXITING FUNCTION...\n')
            return
        end
    else
       p = TRANSFERback(wd, sid, pqt, pqf, flatwfiles, actualim3num, dd, idx);
       if p ~= 3
           fprintf('ERROR IN PQT. EXITING FUNCTION...\n')
           return
        end
    end
end
%
end
%%
function [pqt,pqf, p2] = getpqt(dd, loc)
%%
% get the process flatw queue table for the drive designated by dd
%
%% Input
% dd = '\\bki05\e$'
%%
%
% open the processing queue on the drive provided
%
titles = {'Machine','Main_Path','Sample','Batch',...
    'Scan','Scan_date','expected_im3','actual_im3','transfer_date',...
    'Flatw_Path','flatfield_binfile_date'...
    'Processing_flatw_Sent','Processing_flatw_Start',...
    'Processing_flatw_Finished','expected_flatw_files','actual_flatw_files'};
%
pqf = [dd,'\Processing_Specimens\process_flatw_queue.csv'];
p2 = 0;
pqt = [];
%
if loc == 1
    if exist(pqf,'file')
        %
        tt = 0;
        %
        while p2 == 0 && tt < 10
            try
                pqt = readtable(pqf,'Delimiter', ',',...
                    'ReadVariableNames',1,'format',repmat('%s ',1,16));
                p2 = 1;
            catch
            end
            tt = tt + 1;
        end
        %
    else
        
        disp(['Processing_queue.csv not found on ', pqf]);
        disp('Creating ...');
        pqt = cell2table(cell(1,16), 'VariableNames', titles);
        writetable(pqt, pqf);
        p2 = 1;
        return
    end
elseif loc == 2
    %
    tt = 0;
    %
    while p2 == 0 && tt < 10
        try
            pqt = readtable(pqf,'Delimiter', ',',...
                'ReadVariableNames',1,'format',repmat('%s ',1,16));
            p2 = 1;
        catch
        end
        tt = tt + 1;
    end
    %
end
%
end
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
%%
function SETUPdir(wd, dd, sid, ScanNum, Scanpath, BatchID)
%%
% set up the directory on bki05 so that we can flat field the case
%
%%
%
% copy over im3 files
%
sor = [Scanpath,'MSI\*.im3'];
des = [dd,'\Processing_Specimens\Specimen\',sid,...
    '\im3\Scan',num2str(ScanNum),'\MSI'];
copyfls(sor,des);
%
% copy BatchID
%
sor = [Scanpath,'BatchID.txt'];
des = [dd,'\Processing_Specimens\Specimen\',sid,'\im3\Scan',num2str(ScanNum)];
copyfile(sor,des);
%
% copy flat field
%
sor = [wd,'\flatfield\flatfield_BatchID_',BatchID,'.bin'];
des = [dd,'\Processing_Specimens\Specimen\flatfield'];
if ~exist(des,'dir')
    mkdir(des)
end
copyfile(sor,des);
%
end
%%
function [C, p] = copyfls(sor,des)
%%
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
%
if isempty(gcp('nocreate'))
    try
        evalc('parpool("local",4)');
    catch
    end
end
%
if ~contains(sor, '\*.im3')
    sorfiles = dir([sor,'\*.im3']);
    if isempty(sorfiles)
        sorfiles = dir([sor,'\*.fw']);
        sorfiles = [sorfiles;dir([sor,'\*.fw01'])];
    end
else
    sorfiles = dir(sor);
end
%
nsor = num2str(length(sorfiles));
bsor = num2str(sum([sorfiles(:).bytes]));
fprintf(['   ',datestr(datetime),'\n'])
sor = replace(sor,'\','\\');
fprintf(['   im3 path "',sor,'"\n'])
fprintf(['      ',nsor,' File(s) ',bsor,' bytes\n'])
%
sor = replace(sor,'\\','\');
cfiles = dir(sor);
ii = strcmp({cfiles(:).name}, '.')|strcmp({cfiles(:).name},'..');
cfiles = cfiles(~ii);
C = ~isempty(cfiles);
if C
    %
    if ~exist(des,'dir')
        mkdir(des)
    end
    %
    parfor i3 = 1:length(cfiles)
        tt = 0;
        p = 0;
        while p == 0 && tt < 10
            sor1 = fullfile(cfiles(i3).folder, cfiles(i3).name);
            try
                copyfile(sor1, des)
                p = 1;
            catch
            end
            if p == 0
                pause(5*60)
                tt = tt + 1
            end
        end
        if p == 0
            fprintf(['WARNING: Failed to copy ',cfiles(i3).name,'\n'])
        end
    end
    %
    desfiles = dir([des,'\*.im3']);
    if isempty(desfiles)
        desfiles = dir([des,'\*.fw']);
        desfiles = [desfiles;dir([des,'\*.fw01'])];
    end
    %
    ndes = num2str(length(desfiles));
    bdes = num2str(sum([desfiles(:).bytes]));
    %
    des = replace(des,'\','\\');
    fprintf(['   dest path "',des,'"\n'])
    fprintf(['      ',ndes,' File(s) ',bdes,' bytes\n'])
    fprintf(['   ',datestr(datetime),'\n'])
    %
    if length(sorfiles) ~= length(desfiles)
        fprintf(['WARNING: not all files transferred properly. DELETING ',...
            'DESTINATION\n   ',des,'\n'])
        try
            des = replace(des,'\\','\');
            rmdir(des,'s')
        catch
        end
        p = 0;
    else 
        p = 1;
    end
end
end
%%
function [p, flatwfiles,actualim3num]...
    = startOneflat(machinename, dd, sid, idx, pqt,ScanNum)
%
% files directory
%
wd = [dd,'\Processing_Specimens\Specimen'];
fwpath = [dd,'\Processing_Specimens\Specimen\flatw'];
%
% get number of actual im3 files
%
actualim3num = str2double(pqt.actual_im3{idx});
v = dir();
flatwcode = fullfile(v(2).folder, v(2).name, 'flatw');
if ~exist(flatwcode, 'dir')
    disp('ERROR: raw2mean_loop worker not set up')
    flatwfiles = 0;
    p = 0;
    return
end
%
flatwpath = [wd, '\', sid, '\im3\flatw'];
try
    rmdir(flatwpath,'s')
catch
end
%
p = 0;
lc = 0;
%
while p == 0 && lc < 10
    %
    % check number of flatwfiles
    %
    actualim3num = dir([wd, '\', sid, '\im3\Scan',...
        num2str(ScanNum),'\MSI\*.im3']);
    actualim3num = length(actualim3num);
    flatwfiles = dir([flatwpath,'\*.im3']);
    flatwfiles = length(flatwfiles);
    %
    % if we dont have the right number of flatw files try code again
    %
    if flatwfiles ~= actualim3num
        if exist(flatwpath,'dir')
            %
            rmdir(flatwpath, 's')
            %
            p2 = [fwpath,'\',sid];
            if exist(p2,'dir')
                rmdir(p2,'s')
            end
        end
        command = [codep,'\Im3Tools\doOneSample ',...
            wd,' ',fwpath,' ',sid,' ',codep];
        % status = system(command); -->> could be used to track errors
        try
            system(command);
        catch
            p = 0;
            return
        end
    else
        %
        % if we do have the right number of files change p to exit loop
        %
        p = 1;
    end
    lc = lc + 1;
end
%
end
%%
function p2 = tryCATCHwritetable(pqt, pqf)
%%
% write a table with error catching. Since multiple functions and users may
% access tables it may not always be allowed to open save over the current
% process tables.
%
%%
    %
    p2 = 0;
    tt = 0;
    wpqf = replace(pqf, '\','\\');
    %
    while p2 == 0 && tt < 10
        try
            writetable(pqt, pqf);
            p2 = 1;
        catch
        end
        if ~p2
            pause(10*60)
            fprintf(['Attempt ',num2str(tt + 1),' to write to ',...
                wpqf,' FAILED...\n   Attempting to transfer again in ',...
                '10 MINUTES...\n'])
            tt = tt + 1;
        end
    end
    %
end
%%
function p = TRANSFERback(wd, sid, pqt, pqf, flatwfiles, actualim3num, dd, idx)
%%
% Transfer the files back to the source locations and
% write out the processing string to the tables. 
%
%%
%
% transfer files flatw im3's
%
des = [wd,'\',sid,'\im3\flatw'];
if exist(des, 'dir')
    try
        rmdir(des,'s')
    catch
        if ~isempty(gcp('nocreate'))
            delete(gcp('nocreate'));
        end
        rmdir(des,'s')
    end
end
%
sor = [dd,'\Processing_Specimens\Specimen\',sid,'\im3\flatw'];
[~, p] = copyfls(sor,des);
fclose('all');
%
% transfer .fw and .fw01 files
%
fwpath = pqt.Flatw_Path{idx};
sor = [dd,'\Processing_Specimens\Specimen\flatw\',sid];
des = [fwpath,'\',sid];
if exist(des, 'dir')
    try
        rmdir(des,'s')
    catch
        if ~isempty(gcp('nocreate'))
            delete(gcp('nocreate'));
        end
        rmdir(des,'s')
    end
end
%
[~, p2] = copyfls(sor,des);
fclose('all');
%
if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
end
%
% remove source locations
%
sor = [dd,'\Processing_Specimens\Specimen\',sid];
try
    rmdir(sor,'s')
catch
end
%
sor = [dd,'\Processing_Specimens\Specimen\flatw\',sid];
try
    rmdir(sor,'s')
catch
end
%
sor = [dd,'\Processing_Specimens\Specimen\flatfield'];
try
    rmdir(sor,'s')
catch
end

p = p + p2;
if p ~= 2
    fprintf('FAILED TO COPY FLATW FILES. EXITING FUNCTION...\n')
    return
end        
%        
pqt.actual_flatw_files(idx) = {num2str(flatwfiles)};
pqt.expected_flatw_files(idx) = {num2str(actualim3num)};
t = datestr(datetime());
t = t(1:11);
pqt.Processing_flatw_Finished{idx} = t;
p2 = tryCATCHwritetable(pqt, pqf);
if p2 == 0
    fprintf('ERROR IN PQT. EXITING FUNCTION...\n')
    return
end
p = p + p2;
%
end