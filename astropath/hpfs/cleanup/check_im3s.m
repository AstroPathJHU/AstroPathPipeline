%% check_im3s
% create by: Benjamin Green
%% -----------------------------------------
% check the bytes and number of im3s for any issues
%
function [err_val, actualim3num] = check_im3s(wd, sname)
%
[Scanpath, ScanNum, ~] = getscan(wd, sname);
%
expectim3num = getAnnotations(Scanpath, ['Scan', num2str(ScanNum)], sname);
im3s = dir([Scanpath,'\MSI\*.im3']);
actualim3num = length(im3s);
%
[err_val] = check_m_files(im3s, expectim3num, 500);
%
end
