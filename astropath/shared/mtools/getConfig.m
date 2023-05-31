function C = getConfig(root,samp,module,varargin)
%%---------------------------------------------------------------
%% getConfig.m
%% Configure C and the log directories (.logpath and .logtop)
%% samp can be an empty string for outer loop processing
%% varargin can be an optional threadid to make sure that top 
%% level logs don't collide if running on multiple threads. 
%% This threadid (if any) will be attached to the end of the name
%% of the log file.
%% 
%% 2020-06-11   Alex Szalay
%%------------------------------------
global logctrl threadid
	%-------------
	% init logctrl
	%-------------
	if (isempty(logctrl))
		logctrl=0;
	end
    %---------------
    % initialize C
    %---------------
    C = [];
    C.err  = 0; 
    C.root = root;
    C.samp = samp;	
    C.module = module;
    if (exist(root)~=7)
        C.err = 1;
        fprintf('ERROR: root path %s not found\n',root);
        return
    end
    %-------------------------
    %% adjust the thread value
    %%------------------------
    C.opt = [];
    if (numel(varargin)>0)
        C.opt = varargin{1};
        threadid = min(C.opt);
    end
    %---------------------------
    % adjust the module name
    %---------------------------
    if (~isempty(threadid))
        C.module = [C.module, sprintf('-%d',threadid)];
    end
    %
    C.dbload = fullfile(C.root,C.samp,'\dbload\');
    C.logdir = fullfile(C.root,C.samp,'logfiles\');
    C.logtop = fullfile(C.root,'\logfiles\');
    %---------------------------------------------
    % create the log paths if they do not exist
    %---------------------------------------------
    if (logctrl>0)
        %
        if (exist(C.logdir)~=7)
            mkdir(C.logdir);
        end
        %
        if (exist(C.logtop)~=7)
            mkdir(C.logtop);
        end
        %
    end
    %
    if (numel(C.samp)==0)
        sep = '';
    else
        sep='-';
    end
    %
    C.project = 0;
    C.cohort  = 0;
    s = getSampledef(C);
    %
    if (isempty(s))
        C.err = 1;
        return
    end
    %
    C.project  = s.Project(1);
    C.cohort   = s.Cohort(1);
    %
    if (numel(strfind(C.samp,'Control'))==0 & numel(C.samp)>0)
        C = getSample(C,s);
    end
    %
    C.logpath = [C.logdir C.samp sep C.module '.log'];
    C.logtop  = [C.logtop C.module '.log'];    
    %-------------------------------------
    % initialize the low level log file
    %-------------------------------------
    if (logctrl>0 & ~isempty(C.samp))
        %
        if (exist(C.logpath)>0)
            delete(C.logpath);
        end    
    %
    end
    %
end


function C = getControl(C)
%%---------------------------------------
%% get the record for a control
%%
%% 2020-06-17   Alex Szalay
%%---------------------------------------
    %
    fprintf('getControl %s\n',C.samp);
    %
    B = getBatch(C);
    C.batch   = s.BatchID;
    C.scannum = s.Scan;
    C.scan    = sprintf('Scan%d',C.scannum);    
end


function C = getSample(C,s)
%%---------------------------------------
%% get the record for a sample
%%
%% 2020-06-17   Alex Szalay
%%---------------------------------------
    %
    %----------------------------------
    % get the corresponding record
    %----------------------------------
    if (numel(C.samp)>0)
        s = s(strcmp(s.SlideID,C.samp)>0,:);
        %
        if (numel(s.Project)==0)
            msg = sprintf('ERROR: cannot find  sample %s',C.samp);
            fprintf('0,0,%s,%s\n',C.samp,msg);
            C.err=1;
            return
        end
        %
        C.batch   = s.BatchID;
        C.scannum = s.Scan;
        C.scan    = sprintf('Scan%d',C.scannum);
    else
        C.batch   = 0;
        C.scan    = 0;
    end
end
