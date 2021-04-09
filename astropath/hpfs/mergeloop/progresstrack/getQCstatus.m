%% QC_done_date
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% checks the status of the inForm_QC.csv file, if there is a date in the
%%% last column then it output at QC_done_date
%% --------------------------------------------------------------------
%%
function QC_done_date = getQCstatus(sname, wd, Targets)
%
% get QC done date from tbl if it exists
% 
QC_done_date = '';
%
wd1 = [wd, '\upkeep_and_progress\inform_QC.csv'];
%
if ~exist(wd1, 'file')
     tbl = array2table(zeros(0,length(Targets) + 4));
     tbl.Properties.VariableNames = [{'Sample'},Targets',...
         {'QC_done_date', 'Initials','Comments'}];
     tbl = [tbl;[sname,repmat({''},1,length(Targets) + 3)]];
     try
        writetable(tbl, wd1)
     catch
         return
     end
else
    tbl = readtable(wd1,'Delimiter',',',...
        'TreatAsEmpty',{' ','#N/A'}, ...
        'Format', (repmat('%s', 1, length(Targets) + 4)));
    %
    try
        tbl1 = tbl(strcmp(tbl.Sample, sname),'QC_done_date');
    catch
        tbl1 = [];
    end
        %
    if ~isempty(tbl1)
        QC_done_date = table2array(tbl1);
    else
        try
            tbl = [tbl; [sname,repmat({''},1,length(Targets) + 3)]];
            writetable(tbl, wd1)
        catch
            try
                tbl.QC_done_date = repmat({''},height(tbl), 1);
                tbl = [tbl; [sname,repmat({''},1,length(Targets) + 1)]];
                writetable(tbl, wd1)
            catch
                return
            end
        end
    end
end
%
end