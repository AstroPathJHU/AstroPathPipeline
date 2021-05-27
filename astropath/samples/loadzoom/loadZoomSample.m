function C=loadZoomSample(root,samp,varargin)
%%-----------------------------------------------
%% execute the sql zoom loading workflow script on one sample
%%
%% 2020-12-15   Alex Szalay
%%-----------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    C = getConfig(root,samp,'loadzoom');
    logMsg(C,'loadZoomSample started',1);
    if (C.err==1)
        return
    end
    %------------------------------
    % get the sampleid, etc
    %------------------------------
    s = getSampledef(C);
    s = s(strcmp(s.SlideID,C.samp)>0,:);
    C.sampleid = s.SampleID(1);
    C.sqlpath = '\\bki02\C\BKI\sql\Zoom\';
    %
    if (opt==-1)
        return
    end
    %-----------------------------------
    % get the workflow definition file
    %-----------------------------------
    try
        wfname = 'zoom-workflow.csv';
        st  = readtable(wfname);
    catch
        msg = sprintf('ERROR: %s not found',wfname);
        logMsg(C,msg,1);
        return
    end
    %
    for j=1:numel(st.stepid)
        %        
        if (strcmp(st.database{j},'sample')>0)
            flag = 1;
            dbname = C.samp;
        elseif (strcmp(st.database{j},'core')>0)
            flag = 0;
            dbname = 'WSICore';
        else
            msg = sprintf('ERROR: Illegal dbname %s',dbname);
            logMsg(C,msg,1);
            return
        end
        %-------------------
        % build the commandccccc
        %-------------------
        if (flag==0)
            sampleid = sprintf('%d',C.sampleid);
            script = replace(st.script{j},'<sampleid>',sampleid);
            script = [' -Q"',script,'"'];
        elseif (flag==1)
            script = [' -b -i',fullfile(C.sqlpath,st.script{j})];
        end
        %
        cmd = ['sqlcmd -Sbki01 -v project=',...
            sprintf('%d',C.project),' -b -C -d',dbname,script];
        %fprintf('%s\n',cmd);
        %-------------------------------
        % now execute the SQL command
        %-------------------------------
        %
        [status,cmdout] = system(cmd);
        %
        if (status>0)
            errorOutput(C,cmdout);
            msg = sprintf('ERROR: loadZoomSample failed');
            logMsg(C,msg,1);
            return
        end
        %----------------------------------------
        % convert the sql output to log messages
        %----------------------------------------
        sqlOutput(C,cmdout);
        %
    end
    %
    logMsg(C,'loadZoomSample finished',1);    
    %
end


function sqlOutput(C,cmdout)
%%---------------------------------------------------------
%% convert the log outputs from the SQL Server scripts
%% into our standard log format.
%%
%% 2020-07-22   Alex Szalay
%%---------------------------------------------------------
    %
    a = splitlines(cmdout);
    %-----------------------------------------------------
    % loop through each line and trim extra white space
    %-----------------------------------------------------
    for i=1:numel(a)
        s = strtrim(a{i});
        if (~isempty(s))
            %fprintf('%s\n',a{1});
            logSql(C,s);
        end
    end
    %
end


function errorOutput(C,cmdout)
%%---------------------------------------------------------
%% convert the error outputs from the SQL Server scripts
%% into our standard log format.
%%
%% 2020-07-22   Alex Szalay
%%---------------------------------------------------------
    %
    a = splitlines(cmdout);
    %-----------------------------------------------------
    % loop through each line and trim extra white space
    %-----------------------------------------------------
    for i=1:numel(a)
        s = strtrim(a{i});
        %
        if (~isempty(s))
            if (~any(ismember(s,';')))
                s = [s,';',sprintf('%s',...
                    datetime('now','Format','yyyy-MM-dd HH:mm:ss'))];
            end
            logSql(C,s);
        end
    end
    %
end

