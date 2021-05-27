function T = getCtrlInfo(C)
%%------------------------------------------------
%% get the information frome TMA image files
%%
%% 2020-08-04   Alex Szalay
%%------------------------------------------------
    %
    for i=1:numel(C.d)
        a = split(C.d(i).name,'_');
        f = fullfile(C.d(i).folder, C.d(i).name);
        tma(i)   = str2num(a{3});
        ctrl(i)  = str2num(a{4});
        date{i}  = a{5};
        name{i}  = C.d(i).name;        
        scan{i}  = getScan(C.root, C.d(i).name);
        batch(i) = getBatchId(f,scan{i});
    end
    %
    prj = 0*tma+C.project;
    coh = 0*tma+C.cohort;
    cid = (1:numel(tma));
    T = table(prj', coh',cid',tma',ctrl',date',batch',scan',name');
    T.Properties.VariableNames = {'Project','Cohort','CtrlID','TMA','Ctrl','Date',...
        'BatchID','Scan','SlideID'};
    %
    if (C.opt==0)
        fname = fullfile(C.dbload,...
            sprintf('project%d_ctrlsamples.csv',C.project));
        writetable(T,fname);
    end
    %    
end



function b = getBatchId(f,scan)
%%----------------------------------------------
%% get the batchid from the BatchId.txt file
%%----------------------------------------------
    %
    b=0;
    ff = fullfile(f,'im3',scan,'BatchID.txt');
    a  = readtable(ff);
    b =  a.Var1(1);
    %
end
