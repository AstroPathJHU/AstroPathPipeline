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
function err_val = mywritetolog(wd, sname, logstring, err_str, locs, tasktype, version)
%
logp = [wd,'\Phenotyped\Results\', tasktype];
err_val = 0;
%
if locs == 1
    %
    if exist(logp,'dir')
        %try
            if strcmpi(tasktype, 'Tables')
                delete([logp,'\*'])
            else
                rmdir(logp, 's')
                mkdir(logp)
            end
        %catch
        %    error(['ERROR IN removing path - TASK:', wd,' ', sname, ' PATH:', logp ]);
        %end
    else
        mkdir(logp)
    end
    %
    if strcmp(tasktype,'Tables') 
        if exist([wd,'\Phenotyped\Results\tmp_ForFiguresTables'],'dir')
            try
                delete([wd,'\Phenotyped\Results\tmp_ForFiguresTables\*'])
            catch
                error(['ERROR IN Results path:', wd,' ', sname, ' ne:', logp]);
            end
        else
            mkdir (wd,'\Phenotyped\Results\tmp_ForFiguresTables')
        end
    end
    %
    % create first line of file
    %
    str = ['MaSS-', tasktype, ' protocol started-v',version, '\n'];
    %
    fprintf(str);
    %
end
%
% for error or warning messages write the message out in the correct format
%
if locs == 2
    %
    if strcmpi(err_str, 'Error:')
        error(err_str);
    end
    %
    fprintf([err_str, '\n'])
    %
end
%
end