%%check_merge_columns
% check that the batch column variables are the correct formatting before
% exporting
function [err_val, b] = check_merge_columns(b, i1)
%
err_val = 0;
%
if isa(b.Opal,'double')
   %
   % if b.Opal is a 'double' convert to a string 
   %
   tmpopal = num2cell(b.Opal);
   tmpopal = cellfun(@(x) num2str(x), tmpopal, 'Uni', 0);
   ii = strcmp(tmpopal, 'NaN');
   %
   if sum(ii) > 1
      ii = find(ii,1);
   end
   %
   tmpopal(ii) = {'DAPI'};
   ss = size(tmpopal);
   if ss(1) == 1
       b.Opal = tmpopal';
   else
       b.Opal = tmpopal;
   end
end
%
if ~isa(b.Opal, 'cell')
  fprintf(['WARNING: batch ', num2str(i1), 'batch file error: column Opal\n'])
  err_val = 1;
  return
end
%
if ~iscell(b.Target)
    fprintf(['WARNING: batch ', num2str(i1), 'batch file error: column Target\n'])
    err_val = 2;
    return
end
%
if ~iscell(b.TargetType)
    fprintf(['WARNING: batch ', num2str(i1), 'batch file error: column TargeType\n'])
    err_val = 3;
    return
end
%
if ~iscell(b.Compartment)
    fprintf(['WARNING: batch ', num2str(i1), 'batch file error: column Compartment\n'])
    err_val = 4;
    return
end
%
% check the data type for the coexpression status column
%
if isa(b.CoexpressionStatus,'double')
   %
   % if b.Opal is a 'double' convert to a string 
   %
   tmpCS = num2cell(b.CoexpressionStatus);
   tmpCS = cellfun(@(x) num2str(x), tmpCS, 'Uni', 0);
   %
   for i1 = 1:length(tmpCS)
       tmpCS_n = tmpCS{i1};
       if length(tmpCS_n) > 3
           ii = 3:3:length(tmpCS_n) - 1;
           t(1:length(tmpCS_n)) = char(0);
           t(ii) = ',';
           tmpCS_n = [tmpCS_n;t];
           tmpCS_n = reshape(tmpCS_n(tmpCS_n ~= 0),1,[]);
           tmpCS{i1} = tmpCS_n;
       end
   end
   %
   b.CoexpressionStatus = tmpCS;
   %
end
%
b.CoexpressionStatus = cellfun(@(x) replace(x, ',',''),...
      b.CoexpressionStatus, 'Uni',0);
%
% check the last 3 columns are all set as numeric
%
SS = b.SegmentationStatus;
if iscell(SS)
    %SS = cell2mat(SS);
    b.SegmentationStatus = str2double(SS);
end
%
SH = b.SegmentationHierarchy;
if ~iscell(SH)
    SH = num2str(SH);
    SH = cellstr(SH);
    b.SegmentationHierarchy = SH;
end
%
SS = b.NumberofSegmentations;
if iscell(SS)
    %SS = cell2mat(SS);
    b.NumberofSegmentations = str2double(SS);
end
%
if ~iscell(b.ImageQA)
    fprintf(['WARNING: batch ', num2str(i1), 'batch file error: column ImageQA\n'])
    err_val = 9;
    return
end
%
if ~iscell(b.Colors)
    fprintf(['WARNING: batch ', num2str(i1), 'batch file error: column Colors\n'])
    err_val = 10;
    return
end
%
end