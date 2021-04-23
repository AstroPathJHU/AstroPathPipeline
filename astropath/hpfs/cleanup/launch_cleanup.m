%% launch_cleanup
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% 
%% --------------------------------------------------------------
%%
function launch_cleanup(main)
%
try
    tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
        'ReadVariableNames', true);
catch
    pause(10)
    tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
        'ReadVariableNames', true);
end
%
for i1 = 1:height(tbl)
    %
    % Clinical_Specimen folder
    %
    td = tbl(i1,:);
    wd = ['\\', td.Dpath{1},'\', td.Dname{1}];
    fw = ['\\', td.FWpath{1}];

end
end