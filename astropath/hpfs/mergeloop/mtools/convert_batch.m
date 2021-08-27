%% convert_batch
% conver the mergeconfig and batch files to the csv file format for
% database injestion. Check the formatting of various columns
%% usage
% convert_batch(main, wd)
% main = '\\bki04\astropath_processing'
% wd = '\\bki04\Clinical_Specimen_7'
function convert_batch(main, wd)
%
try
    cohorts_progress = readtable(fullfile(main, 'AstropathCohortsProgress.csv'));
catch
    fprintf('ERROR: could not open AstropathCohortsProgress.csv \n')
    return
end
%
bases = strcat(repmat({'\\'}, height(cohorts_progress), 1), ...
    cohorts_progress.Dpath, repmat({'\'},...
    height(cohorts_progress), 1),...
    cohorts_progress.Dname);

ii = contains(bases, wd);
%
cohort = cohorts_progress.Cohort(ii);
project = cohorts_progress.Project(ii);
%
batches = dir(fullfile(wd, 'Batch\Batch*.xlsx'));
if isempty(batches)
    fprintf('ERROR: No batch files in Batch directory \n')
end
%
for i1 = 1:length(batches)
    batch = fullfile(batches(i1).folder, batches(i1).name);
    %
    try
        b = readtable(batch);
    catch
        fprintf(['WARNING: Could not open batch ', num2str(i1), ' Batch file \n'])
        continue
    end
    %
    if width(b) == 9
        bnames = b.Properties.VariableNames;
        b = [array2table([repmat(project, height(b), 1), repmat(cohort, height(b), 1)]), b];
        b.Properties.VariableNames = ['Project','Cohort', bnames];
        %
        [err_val, b] = check_batch_columns(b, i1);
        if err_val ~= 0
            continue
        end
        %
        bout = replace(batch, 'xlsx', 'csv');
        try
            writetable(b, bout);
        catch
            fprintf(['WARNING: Could not write new batch ', num2str(i1), ' Batch file \n'])
        end
        %
    else
        fprintf(['WARNING: batch ', num2str(i1), ' Batch file did not have 9 columns\n'])
    end
    %
    % do the merge config file
    %
    merge = replace(batch, 'BatchID','MergeConfig');
    %
    try
        m = readtable(merge);
    catch
        fprintf(['WARNING: Could not open batch ', num2str(i1), ' MergeConfig file \n'])
        continue
    end
    %
    if width(m) == 11
        mnames = m.Properties.VariableNames;
        m = [array2table([repmat(project, height(m), 1),...
            repmat(cohort, height(m), 1), m.BatchID, ...
            (1:height(m))']), m(:, 2:end)];
        m.Properties.VariableNames = ['Project','Cohort', 'BatchID', 'layer', mnames(2:end)];
        %
        [err_val, m] = check_merge_columns(m, i1);
        %
        m = add_af(m, mnames);
        m.Colors = [];
        %
        if err_val ~= 0
            continue
        end
        %
        mout = replace(merge, 'xlsx', 'csv');
        try
            writetable(m, mout);
        catch
            fprintf(['WARNING: Could not write new batch ', num2str(i1), ' MergeConfig file \n'])
        end
        %
    else
       fprintf(['WARNING: batch ', num2str(i1), ' MergeConfig file did not have 11 columns\n'])
    end
    %
end
end