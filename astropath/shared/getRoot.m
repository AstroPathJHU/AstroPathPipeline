function root = getRoot(project)
%%-------------------------------------------------------------
%% get the root directory corresponding to the project id
%% Alex Szalay, 2020-10-30
%%------------------------------------------------------------
    %
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
        root = [];
        return
    end    
    %
end