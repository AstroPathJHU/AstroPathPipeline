%% check_fw_fw01
% create by: Benjamin Green
%% -----------------------------------------
% check the bytes and number of fw and fw01s for any issues
%
function [err_val] = check_fw_fw01(wd, sname, expectim3num)
%
fws = dir([wd,'\', sname, '\*.fw']);
fw01s = dir([wd,'\', sname, '\*.fw01']);
%
[err_val] = check_m_files(fws, expectim3num, 500);
[err_val2] = check_m_files(fw01s, expectim3num, 200);
%
err_val = err_val + err_val2;
%
end
