%% Progress Tracker for Clinical_Specimens on bki04
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% For all Clinical Specimens in the folder track progress if the samples
%%% are ready to be moved forward this function will call the necessary functions
%% --------------------------------------------------------------
%%
function [] = progresstrack_samples_loop(main, wd, machine, logstring)
%
%start by getting all the folder names
%
samplenames = find_specimens(wd);
[e_val, tmpfd, Targets] =  intialize_folders(wd);
if e_val > 0
    return
end
%
[ss, ssname, AB] = initialize_samples_spreadsheet(wd, tmpfd, samplenames);
%
%initialize all tracking vectors
%
[samplenamesout, BatchID, ScanNum, transferdate, actualim3num, ...
    expectim3num, errorim3num, Scandate, actualflatwnum, ...
    expectflatwnum, errorflatwnum, flatwdate, actual_infm, ...
    expect_infm, diff_infm, infmd, algd, diff_ifall, aifall, exifall, ...
    ifalldate, actual_merged_tables, expect_merged_tables, ...
    diff_merged_tables, MergeTblsDate, QCImagesdate, QCImages, ...
    QC_done_date] = intialize_vars(tmpfd);
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
        [~,idx] =  max(datenum(infmda));
        difallfd = infmda{idx};
        ifalldate{i1} = iffda(idx).date(1:11);
    end
    MergeConfig = [wd,'\Batch\MergeConfig_',BatchID{i1},'.xlsx'];
    %
    % get merge tables info and run MaSS if needed
    %
    [MergeTbls, MergeTblsDate{i1}, Rfd] = getmergefiles(...
        sname, informpath, trackinform, tmpfd,...
        difallfd,expectedTablesnum, MergeConfig, logstring);
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
        getQCfiles(sname, informpath, Rfd, MergeConfig, logstring);
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