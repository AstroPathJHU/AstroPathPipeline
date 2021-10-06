%% check_tables
% create by: Benjamin Green
%% -----------------------------------------
% check the bytes and number of tables for any issues
%
function [err_val] = check_tables(wd, sname, expectim3num)
%
tbls = dir([wd,'\', sname, '\inform_data\Phenotyped\Results\Tables\*.csv']);
[err_val] = check_m_files(tbls, expectim3num, -1);
%
end
