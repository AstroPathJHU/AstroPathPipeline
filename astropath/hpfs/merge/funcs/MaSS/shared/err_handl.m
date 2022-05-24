
%% function: err_handl
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 07/07/2020
%% --------------------------------------------------------------
%% Description
% if there is an err_val that is not 0; output the corresponding message
% and return and exit code 1 to exit MaSS (errors) 0 to keep going
% (warnings)
%% --------------------------------------------------------
%%
function e_code = err_handl(wd, sname, logstring, Markers, err_val, tasktype)
%
if strcmp(tasktype, 'Tables')
    e_code = err_handl_tables(wd, sname, logstring, Markers, err_val);
else
    e_code = err_handl_qaqc(wd, sname, logstring, Markers, err_val);
end
%
end
