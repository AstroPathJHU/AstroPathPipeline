function [ss, ssname, AB] = initialize_samples_spreadsheet(wd, tmpfd, samplenames)

%
% the name of the summary sheet
%
ssname = [wd,'\upkeep_and_progress\samples_summary.xlsx'];
%
%fill the name vector of the ABs used
%
track = 1;
AB = cell(length(tmpfd)*4,1);
for i1 = 1:length(tmpfd)
    AB{track} = [tmpfd(i1).name,'_Expected_InForm_Files'];
    AB{track+1} = [tmpfd(i1).name,'_Actual_InForm_Files'];
    AB{track+2} = [tmpfd(i1).name,'_Errors_InForm_Files'];
    AB{track+3} = [tmpfd(i1).name,'_InForm_Algorithm'];
    AB{track+4} = [tmpfd(i1).name,'_InForm_date'];
    track = track + 5;
end
%
% create summary spreadsheet
%
tblsz = 25 + length(AB);
ss = array2table(zeros(0,tblsz));
ss.Properties.VariableNames = [{'Machine','Main_Path','Sample','Batch',...
    'Scan','Scan_date','Expected_Im3s','Actual_Im3s','Errors_Im3s','Transfer_Date',...
    'Expected_Flatw_Files','Actual_Flatw_Files','Errors_Flatw_Files','Flatw_Date'},AB',...
    {'All_Expected_InForm_Files','All_Actual_InForm_Files',...
    'All_Errors_InForm_Files','All_InForm_Date','Expected_Merged_Tables',...
    'Actual_Merged_Tables','Errors_Merged_Tables','Merge_Tables_Date',...
    'Actual_QC_Images','QC_Ready_Date','QC_Done_Date'}];
%
%populate the table with all blank rows to start.
%
ss = [ss;[repmat({''},length(samplenames),tblsz)]];
end