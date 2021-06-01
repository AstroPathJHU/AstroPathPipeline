%% check_xmls
% create by: Benjamin Green
%% -----------------------------------------
% check the bytes and number of xmls for any issues
%
function [err_val] = check_xmls(wd, sname, expectim3num)
%
xmls = dir([wd,'\', sname, '\im3\xml\*.Exposure.xml']);
[err_val] = check_m_files(xmls, expectim3num, 300);
%
xmls = dir([wd,'\', sname, '\im3\xml\*.Parameters.xml']);
[err_val2] = check_m_files(xmls, 1, -1);
%
xmls = dir([wd,'\', sname, '\im3\xml\*.Full.xml']);
[err_val3] = check_m_files(xmls, 1, -1);
%
err_val = err_val + err_val2 + err_val3;
%
end
