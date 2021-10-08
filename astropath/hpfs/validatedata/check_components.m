%% check_components
% create by: Benjamin Green
%% -----------------------------------------
% check the bytes and number of components for any issues
%
function [err_val] = check_components(wd, sname, expectim3num)
%
comps = dir([wd,'\', sname, '\inform_data\Component_Tiffs\*data.tif']);
[err_val] = check_m_files(comps, expectim3num, -1);
%
comps = dir([wd,'\', sname, '\inform_data\Component_Tiffs\*data_w_seg.tif']);
[err_val2] = check_m_files(comps, expectim3num, -1);
%
err_val = err_val + err_val2;
%
end
