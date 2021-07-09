function [ii2, tbl3, fnms] = checkMean(wd, B1, tbl2)
%%
% check if the mean flat field file for each sample exists
% exclude batches with, 'artifact_detected.csv' in the im3 path
%
% B1 = string, current batch number 
% wd = string, current main directory
% tbl2 = table of samples, batchids, and scanpaths for all samplesi
% in wd
%
%%
%
% check if the other .flt files exist
%
ii = strcmp(tbl2.BatchID,B1);
tbl3 = tbl2(ii,:);
%
p = cellfun(@(x)[wd,'\',x,'\im3\*artifact_detected.csv'],tbl3.Sample,'Uni',0);
fnms = cellfun(@(x)dir(x),p,'Uni',0);
ii2 = cellfun(@(x)~isempty(x),fnms,'Uni',0);
ii2 = [ii2{:}];
tbl3(ii2,:) = [];
%
p = cellfun(@(x)[wd,'\',x,'\im3\*mean.flt'],tbl3.Sample,'Uni',0);
fnms = cellfun(@(x)dir(x),p,'Uni',0);
ii2 = cellfun(@(x)isempty(x),fnms,'Uni',0);
ii2 = [ii2{:}];
%
end
%