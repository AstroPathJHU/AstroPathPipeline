function mydate = return_date_max(dd)
%
if ~isempty(dd)
    [~,idx] = max([dd(:).datenum]);
    mydate = dd(idx).date;
else
    mydate = [];
end
%
end