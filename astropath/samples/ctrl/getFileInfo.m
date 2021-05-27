function t = getCtrlInfo(C)
%%------------------------------------------------
%% get the information frome TMA image files
%%
%% 2020-08-04   Alex Szalay
%%------------------------------------------------
    %
    for i=1:numel(C.d)
        a = split(C.d(i).name,'_');
        f = fullfile(C.d(i).folder, C.d(i).name);
        tma(i)= str2num(a{3});
        ctrl(i) = str2num(a{4});
        date{i}= a{5};
        name{i} = C.d(i).name;        
        scan{i} = getScan(C.root, C.d(i).name);
        batch(i) =  getBatchInfo(f,scan{i});
    end
    %
    t = table(tma',ctrl',date',batch',scan',name');
    t.Properties.VariableNames = {'TMA','Ctrl','Date',...
        'BatchID','Scan','Name'};
    %
end



function b = getBatchInfo(f,scan)
%%-----------------------------------
%% get the batchid for this sample
%%
%% 2020-08-04   Alex Szalay
%%-----------------------------------
    %
    g = fullfile(f,'im3',scan,'BatchID.txt');
    t = readtable(g);
    b = t.Var1(1);
    %
end
