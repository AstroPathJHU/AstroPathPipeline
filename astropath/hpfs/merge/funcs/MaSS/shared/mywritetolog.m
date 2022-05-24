% function: mywritetolog
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 07/07/2020
%% --------------------------------------------------------------
%% Description
% create the log and output resultant error messages
%% --------------------------------------------------------------
%% input:
% err_val = exit code value indicating different errors
% loc = location in main code block of log file message
% wd = working directory of current specimen up to inform_data\Phenotyped
% tim = contains different file and time information
%%
function err_val = mywritetolog(wd, sname, logstring, err_str, locs, version, tasktype)
%
tim = datestr(now,'yyyy-mm-dd HH:MM:SS');
logp = [wd,'\Phenotyped\Results\', tasktype];
err_val = 0;
%
if locs == 1
    %
    if exist(logp,'dir')
        try
            delete([logp,'\*'])
        catch
            err_val = 9;
            warning(['ERROR IN path:', wd,' ', sname]);
            return
        end
    else
        mkdir(logp)
    end
    %
    if strcmp(tasktype,'Tables') 
        if exist([wd,'\Phenotyped\Results\tmp_ForFiguresTables'],'dir')
            try
                delete([wd,'\Phenotyped\Results\tmp_ForFiguresTables\*'])
            catch
                err_val = 10;
                warning(['ERROR IN Results path:', wd,' ', sname]);
                return
            end
        else
            mkdir (wd,'\Phenotyped\Results\tmp_ForFiguresTables')
        end
    end
    %
    % create first line of file
    %
    str = [logstring, sname, ';MaSS - ',...
        tasktype, ' protocol started-v',version,';', tim, '\n'];
    %
    fprintf(str);
    %
end
%
% for error or warning messages write the message out in the correct format
%
if locs == 2
    %
    fprintf([logstring, sname, ';',err_str,';', tim, '\n']);
    %
end
%
end