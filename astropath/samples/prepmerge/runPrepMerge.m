function C = runPrepMerge(project,varargin)
%%--------------------------------------------------------
%% convert the xlsx files in the Batch subdirectory
%% to csv, in preparation for loading, and create a 
%% Batch\loadfiles.csv. Only write the output if opt==0.
%% For opt=0 also write the logs
%%
%% 2020-07-29   Alex Szalay
%%--------------------------------------------------------
global logctrl
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    if (opt==0)
        logctrl=1;
    end
    %------------------------------
    % get the cohort definition
    %------------------------------
    f = '\\bki04\astropath_processing';
    g = fullfile(f,'AstroPathCohortsProgress.csv');
    %
    %--------------------------------------------
    % open the cohorts file and get the project
    %--------------------------------------------
    try
        c = readtable(g);
        c = c(c.Project==project,:);
        root = fullfile('\\',c.Dpath{1},c.Dname{1});
    catch
        fprintf('ERROR: could not open %s\n',f);
        return
    end
    %---------------------------------------------------------
    % set the basic params, log path and top level logfile 
    %---------------------------------------------------------
    C = getConfig(root,'','prepmerge');    
    logMsg(C,'runPrepMerge started',1);
    %
    C.opt  = opt;
    C.samp = sprintf('project%d',C.project);
    C.batch    = fullfile(C.root,'Batch');
    C.clinical = fullfile(C.root,'Clinical');
    C.ctrl     = fullfile(C.root,'Ctrl');
    C.logdir   = C.batch;
    C.loadfile = fullfile(C.dbload,[C.samp,'_loadfiles.csv']);
    %-----------------------------------
    % make sure that the dbload exists
    %-----------------------------------
    if (exist(C.dbload)~=7)
        mkdir(C.dbload);
    end    
    %
    C = scanMergeFiles(C);
    %
    logMsg(C,'runPrepMerge finished',1);
    %
    logctrl=0;
    return
    %    
end



