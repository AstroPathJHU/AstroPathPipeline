function b = createSampleFile(proj,varargin)
%%---------------------------------------------------
%% Scan the cohort file for the project.
%% Find the samples and build a samples.csv file
%% sort the table by proper numbering before adding the serial numbers
%%
%% 2020-06-07   Alex Szalay
%%---------------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    f = '\\bki04\astropath_processing';
    g = fullfile(f,'AstroPathCohortsProgress.csv');
    h = fullfile(f,'AstroPathSampledef.csv');
    %
    %--------------------------
    % open the cohorts file
    %--------------------------
    try
        c = readtable(g);
    catch
        fprintf('ERROR: could not open %s\n',f);
        return
    end
    %-----------------------------------------------
    % if proj==0, show list of projects and return
    %-----------------------------------------------
    if (proj==0)
        b = [];
        return
    end
    %----------------------------------------------
    % get the max of the SampleID allocated so far
    %----------------------------------------------
    try
        s = readtable(h);
    catch
        fprintf('ERROR: could not open %s\n',g);
        return
    end
    %
    start = max(s.SampleID)+1;
    fprintf('start(SampleID)=%d\n',start);
    %---------------------------------------
    % get the row for the current project
    %---------------------------------------
    %
    h = c(c.Project==proj,:);
    %
    if (numel(h.Project)>1)
        fprintf('More than one line returned\n');
        return
    end
    %---------------------------------------------
    % parse the directory, keep only the slides
    %---------------------------------------------
    g = ['\\' h.Dpath{1} '\' h.Dname{1} '\'];
    d = dir(g);        
    %------------------------------------------------
    % reject all subdirectories which are not samples
    %------------------------------------------------
    ix = [];
    for i=1:numel(d)
        if(~exist(fullfile(d(i).folder,d(i).name,'im3')))
            ix = [ix,i];
            continue
        end
        %
        if(numel(strfind(d(i).name,'Control'))>0)
            ix = [ix,i];
        end        
    end
    d(ix) = [];
    %--------------------------------------
    % check against existing samples
    %--------------------------------------
    %------------------------------------------
    % get the scan and the batch numbers
    %------------------------------------------
    for i=1:numel(d)
        %
        d(i).snum  = getScanNumber(d(i).folder,d(i).name);
        %
        fname = [d(i).folder,'\',d(i).name,'\im3\',...
            sprintf('Scan%d',d(i).snum),'\BatchID.txt'];        
        try
            bnum = csvread(fname);
        catch
            bnum = 0;
            fprintf('Missing batch file %s \n',fname);
        end
        d(i).batch = bnum;
        %
    end
    %--------------------------
    % loop through the samples
    %--------------------------
    for i = 1:numel(d)
        samp(i) = 0;
        slide{i} = d(i).name;
        cohort(i) = h.Cohort;
        project(i) = h.Project;
        snum(i) = d(i).snum;
        bnum(i) = d(i).batch;
    end
    %slide{15}='M99';
    %slide{16}='M999';
    %---------------------
    % insert into table
    %---------------------
    [q,is] = sort_nat(slide);
    
    b = table(samp(is)',slide(is)',project(is)',cohort(is)',snum(is)',bnum(is)');
    %b = table(samp',slide',project',cohort',snum',bnum');
    b.Properties.VariableNames = {'SampleID','SlideID','Project',...
        'Cohort','Scan','BatchID'};        
    %----------------------------
    % insert the sample numbers
    %----------------------------
    b.SampleID = (start:start+numel(b.Project)-1)';
    b.isGood = 1+ 0*b.SampleID;
    %-------------------
    % save if opt==0
    %-------------------
    if (opt==0)
        f = ['\\bki02\c\BKI\save\'];
        fname = [g,'sampledef.csv'];
        writetable(b,fname);
        fprintf('%s created\n',fname);
    end
    %
end





