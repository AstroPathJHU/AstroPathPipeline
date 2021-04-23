%% clean_cohort
% create by: Benjamin Green
%% -----------------------------------------
% check the bytes and number of files in all samples of a cohort
% for any issues. Then prepare the batch and merge files.
%
function clean_cohort(wd, fw)   
%
% get specimen names for the CS
%
samplenames = find_specimens(wd);
%
% cycle through and search for the xml files
%
out = zeros(length(samplenames), 8);
%
for i2 = 1:length(samplenames)
    %
    sname = samplenames{i2};
    [im3_err_val, expectim3num] = check_im3s(wd, sname);
    [fw_err_val] = check_fw_fw01(fw, sname, expectim3num);
    [flatw_err_val] = check_flatws(wd, sname, expectim3num);
    [xml_err_val] = check_xmls(wd, sname, expectim3num);
    [comps_err_val] = check_components(wd, sname, expectim3num);
    [tbl_err_val] = check_tables(wd, sname, expectim3num);
    %
    vals = [im3_err_val, fw_err_val, flatw_err_val, xml_err_val, comps_err_val, tbl_err_val];
    total_err_val = sum(vals);
    %
    out(i2,:) = [i2, vals, total_err_val];
    %
    fprintf([sname,' Complete\n'])
    %
end
%
tout = array2table(out);
tout.slideid = samplenames';
tout = tout(:, [9,2:8]);
names = {'slideid','im3','fw','flatw','xml','comps','tbl','total'};
tout.Properties.VariableNames = names;
%
loc = [wd,'\upkeep_and_progress\sample_error_codes.csv'];
try
    writetable(tout, loc)
catch
end
%
prepare_merge_batch(wd);
%
end