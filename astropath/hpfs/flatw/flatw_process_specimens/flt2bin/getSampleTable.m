function tbl2 = getSampleTable(wd, samplenames)
%%
% get the table with the scanpath and batchid
% for each sample
%
%%
tbl2 =  cell2table(cell(0,3), 'VariableNames',...
    {'Sample','BatchID','Scanpath'});
%
for i2 = 1:length(samplenames)
    sname = samplenames{i2};
    %
    try
        [Scanpath, ~, BatchID] = getscan(wd, sname);
    catch
        fprintf(['"',sname,'" is not a valid clinical specimen folder \n']);
        continue
    end
    %
    if isempty(BatchID)
        fprintf(['"',sname,'" is not a valid clinical specimen folder \n']);
        continue
    end
    %
    tbl3 = table();
    tbl3.Sample = {sname};
    tbl3.BatchID = {BatchID};
    tbl3.Scanpath = {Scanpath};
    tbl2 = [tbl2;tbl3];
end
%
end