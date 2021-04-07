%% progresstrack_cohorts_loop 
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
function progresstrack_cohorts_loop(main)
%
tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
tbl2 = readtable([main, '\AstropathConfig.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
tbl3 = readtable([main, '\AstropathCohortsProgress.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
%
addpath("MaSS", "progresstrack")
%
for i1 = 1:height(tbl)
    %
    % Clinical_Specimen folder
    %
    td = tbl(i1,:);
    wd = ['\\', td.Dpath{1},'\', td.Dname{1}];
    project = td.Project;
    %
    % check drive space
    %
    str = ['\\', td.Dpath{1}];
    TB = java.io.File(str).getFreeSpace * (10^-12);
    TB = round(TB, 3);
    %
    tbl2(tbl2.Project == project,'Space_TB') = {TB};
    %
    try
        writetable(tbl2,[main, '\AstropathConfig.csv'])
    catch
    end
    %
    % run progress tracker unless process string is No from paths file
    %
    gostr = tbl2(tbl2.Project == project,'Process_Merge');
    gostr = table2array(gostr);
    gostr = gostr{1};
    if strcmpi(gostr,'No')
        continue
    end
    %
    cohort = table2array(tbl3(tbl3.Project == project,'Cohort'));
    logstring = sprintf('%d;%d;', project, cohort);
    machine = tbl3(tbl3.Project == project,'Machine');
    %
    progresstrack_samples_loop(main, wd, machine, logstring);
    %
    % fill main inForm queue
    %
    try
        pop_main_queue(wd, main)
    catch
    end
end
%
rmpath("MaSS", "progresstrack")
%
end