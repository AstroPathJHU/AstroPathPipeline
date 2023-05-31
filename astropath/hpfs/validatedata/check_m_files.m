%% check_m_files
% create by: Benjamin Green
%% -----------------------------------------
% check the bytes and number of files for any issues
%
function [err_val] = check_m_files(files, expected, brange)
%
err_val = 0;
%
filesnum = length(files);
%
fileserrs = filesnum - expected;
%
if fileserrs ~= 0
   err_val = err_val + 1;
end
%
% check bytes
%
bytes = {files(:).bytes};
%
if brange > 0
    if range(cell2mat(bytes)) > brange
        err_val = err_val + 1;
    end
else
    ii = any(cell2mat(bytes) == 0);
    err_val = err_val + ii;
end
%
end
