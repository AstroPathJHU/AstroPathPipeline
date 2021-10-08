%%check_batch_columns
% check that the batch column variables are the correct formatting before
% exporting
function [err_val, b] = check_batch_columns(b, i1)
%
err_val = 0;
%
if ~iscell(b.OpalLot)
    fprintf(['WARNING: batch ', num2str(i1), 'Batch file error: column OpalLot\n'])
    err_val = 1;
    return
end
%
if ~iscell(b.OpalDilution)
    fprintf(['WARNING: batch ', num2str(i1), 'Batch file error: column OpalDilution\n'])
    err_val = 2;
    return
end
%
if isa(b.Opal,'double')
   %
   % if B.Opal is a 'double' convert to a string 
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
  fprintf(['WARNING: batch ', num2str(i1), 'Batch file error: column Opal\n'])
  err_val = 3;
  return
end
%
if ~iscell(b.Target)
    fprintf(['WARNING: batch ', num2str(i1), 'Batch file error: column Target\n'])
    err_val = 4;
    return
end
%
if ~iscell(b.Target)
    fprintf(['WARNING: batch ', num2str(i1), 'Batch file error: column Target\n'])
    err_val = 4;
    return
end
%
if ~iscell(b.Compartment)
    fprintf(['WARNING: batch ', num2str(i1), 'Batch file error: column Compartment\n'])
    err_val = 5;
    return
end
%
if ~iscell(b.AbLot)
    fprintf(['WARNING: batch ', num2str(i1), 'Batch file error: column AbLot\n'])
    err_val = 6;
    return
end
%
if ~iscell(b.AbDilution)
    fprintf(['WARNING: batch ', num2str(i1), 'Batch file error: column AbDilution\n'])
    err_val = 7;
    return
end
%
end