function Z = runQueue(project,module,varargin)
%%--------------------------------------------------
%% Run a queue manager on one <project>, executing <module>
%% project: number specifying the project number in the cohor progrss file
%% module : short name of the package we want to execute
%% varargin: optional parameters
%% Returns the dstruct with the set of samples
%% Creates a queue-<module.csv file in the logfiles directory at the root
%% Example:
%%      d = runQueue(13,'prepdb')   -- run prepdb on the validation cohort
%%      d = runQueue(11,'shredxml')
%% -------------------------------------------------
%% 2020-07-02   Alex Szalay  built the queue based generic harness 
%%--------------------------------------------------
global logctrl
	%------------------------------------
	% go to the directory of the module
	%------------------------------------
	codepath = '\\bki02\C\BKI\matlab';
	workdir  = fullfile(codepath,module);
	if (exist(workdir)~=7)
		fprintf('Working directory %s not found\n',workdir);
		return
	else
		cd(workdir);
	end
	%
    logctrl=1;
    C=[];
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
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
    Z = getConfig(root,'',module);    
    %----------------------
    % get sampledef file
    %----------------------
    Z.T = getSampledef(Z);
    %--------------
    % get option
    %--------------
    Z.opt = [];
    if (numel(varargin)>0)
        Z.opt = varargin{1};
    end
    %--------------------------
    % initialize the QUEUE
    %--------------------------
    Z.queue = fullfile(Z.logdir,[Z.module,'-queue.csv']);
    if (exist(Z.queue)~=2)
        %
        fp = fopen(Z.queue,'w');
        fprintf(fp,['SampleID,SlideID,ServerName,',...
            'Status\n0,NULL,NULL,none\n']);
        fclose(fp);
        %----------------------------------------
        % delete detailed logs, if in write mode
        %----------------------------------------
        if (logctrl>0 & exist(Z.logtop)==2)
            delete(Z.logtop);
            fprintf('deleting logfile %s\n',Z.logtop);
        end
        %
    else
        try
            Q = readtable(Z.queue);
        catch
            logMsg(Z,'ERROR: cannot read queue file');
            return
        end
        if (max(Q.SampleID) == max(Z.T.SampleID))
            fprintf('Queue has completed.\nTo restart, delete %s\n',...
                Z.queue);
        end
        %
    end
    %------------------------------
    % start loop until we reach
    % the end of samples
    %------------------------------
    while(1==1)
        try
            Q = readtable(Z.queue);
        catch
            logMsg(Z,'ERROR: cannot read queue file');
        end
        %-----------------------------------------
        % loop through the samples with isGood==1
        % break, if there are no more samples
        %-----------------------------------------
        maxsamp = max(Q.SampleID);
        this = Z.T(Z.T.SampleID>maxsamp & Z.T.isGood==1,'SampleID');   
        %
        if (isempty(this))
            break            
        end
        %
        n = min(this.SampleID);
        t = Z.T(Z.T.SampleID==n,:);
        samp = t.SlideID{1};        
        %-----------------------
        % update the queue file
        %-----------------------
        fp = fopen(Z.queue,'a');
        fprintf(fp,'%d,%s,%s,%s\n',n,samp,...
            getenv('computername'),'start');
        fclose(fp);
        %====================================
        % execute the core function
        %------------------------------------
        fprintf('%s, %s, %s\n',samp, module,datestr(datetime('now')));
		%
        try 
            if (strcmp(module,'prepdb'))
                C = prepSample(Z.root,samp);
            elseif (strcmp(module,'shredxml'))
                C = shredXML(Z.root,samp);
            elseif (strcmp(module,'zoom'))
                C = runZoom(Z.root,samp);
            elseif (strcmp(module,'annowarp'))
                C = runAnnowarp(Z.root,samp);
            elseif (strcmp(module,'geomcell'))
                C = geomCells(Z.root,samp);
            elseif (strcmp(module,'geom'))
                C = geomSample(Z.root,samp);
            elseif (strcmp(module,'csvscan'))
                C = scanCells(Z.root,samp);
            elseif (strcmp(module,'loaddb'))
                C = loadSampleDB(Z.root,samp);
            elseif (strcmp(module,'deepzoom'))
                C = runDeepZoomSample(Z.root,samp);	
            elseif (strcmp(module,'loadzoom'))
                C = loadZoomSample(Z.root,samp);           
            end
            C.err = 0;
        catch
            msg = sprintf('ERROR: %s FAILED',module);
            logMsg(Z,msg,1);           
        end
        %====================================
        if (isempty(C))
            continue
        end
        %--------------------
        % update the queue file
        %--------------------
        if (C.err==1)
            %
            msg = sprintf('%s FAILED',module);
            logMsg(C,msg,1);   
            %
            fp = fopen(Z.queue,'a');
            fprintf(fp,'%d,%s,%s,%s\n',n,samp,...
                getenv('computername'),'FAILED');
            fclose(fp); 
        else
            fp = fopen(Z.queue,'a');
            fprintf(fp,'%d,%s,%s,%s\n',n,samp,...
                getenv('computername'),'finish');
            fclose(fp);
        end
    end
    %----------------------
    % disable hard logging
    %----------------------
    logctrl=0;
    %
end
