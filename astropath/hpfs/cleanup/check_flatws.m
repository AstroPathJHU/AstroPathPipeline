%% check_flatws
% create by: Benjamin Green
%% -----------------------------------------
% check the bytes and number of flatws for any issues
%
function [err_val] = check_flatws(wd, sname, expectim3num)
%
flatws = dir([wd,'\', sname, '\im3\flatw\*.im3']);
[err_val] = check_m_files(flatws, expectim3num, 500);
%
end
