function [samplenamesout, BatchID, ScanNum, transferdate, actualim3num, ...
            expectim3num, errorim3num, Scandate, actualflatwnum, expectflatwnum, errorflatwnum, flatwdate,...
            actual_infm, expect_infm, diff_infm, infmd, algd, diff_ifall,...
            aifall, exifall, ifalldate, actual_merged_tables, expect_merged_tables,...
            diff_merged_tables, MergeTblsDate, QCImagesdate, QCImages, QC_done_date]...
            = intialize_vars(tmpfd)
samplenamesout = cell(1);
BatchID = cell(1);
ScanNum = cell(1);
transferdate = cell(1);
actualim3num = cell(1);
expectim3num = cell(1);
errorim3num = cell(1);
Scandate = cell(1);
actualflatwnum = cell(1);
expectflatwnum = cell(1);
errorflatwnum = cell(1);
flatwdate = cell(1);
actual_infm = cell(1,length(tmpfd));
expect_infm = cell(1,length(tmpfd));
diff_infm = cell(1,length(tmpfd));
infmd = cell(1,length(tmpfd));
algd = cell(1,length(tmpfd));
diff_ifall =  cell(1);
aifall =  cell(1);
exifall =  cell(1);
ifalldate =  cell(1);
actual_merged_tables =  cell(1);
expect_merged_tables =  cell(1);
diff_merged_tables = cell(1);
MergeTblsDate =  cell(1);
QCImagesdate = cell(1);
QCImages = cell(1);
QC_done_date = cell(1);
end