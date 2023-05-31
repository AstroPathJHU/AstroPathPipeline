function logSql(C,msg,varargin)
%%--------------------------------------------
%% Write log message to the console or to a logfile,
%% depending on the value of the global variable
%% the varargin==1 means critical messages
%% The message comes from SQL, together with the sample name
%% and with its own timestamp.
%%
%% Alex Szalay, Baltimore, 2019-02-03
%%-------------------------------------------
global logctrl
    critical = 0;
    if (numel(varargin)>0)
        critical = varargin{1};
    end
    %
    s = sprintf('%d;%d;%s;%s',...
        C.project,C.cohort,C.samp,msg);
    %
    if (logctrl>0)
        %
        dlmwrite(C.logpath,s,'delimiter','','newline','pc','-append');
        %
        % critical messages also go to central log
        %
        if (critical==1)
            dlmwrite(C.logtop,s,'delimiter','','newline','pc','-append');
        end
        %
    else
        fprintf('%s\n',s);
    end
    %
end