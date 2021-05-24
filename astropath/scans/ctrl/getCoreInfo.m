function t = getCoreInfo(C)
%%--------------------------------------------------
%% get the tissue information from the Excel file
%% and convert it to a more parseable format
%%
%% 2020-08-04   Alex Szalay
%%--------------------------------------------------
    %
    xlsx = fullfile(C.root,'Ctrl','Control_TMA_info.xlsx');
    if (~exist(xlsx))
        C.err=1;
        msg = 'ERROR: Control_TMA_info.xlsx file not found';
        logMsg(C,msg,1);
        return
    end
    %
    try
        a = readtable(xlsx);
    catch
        C.err=1;
        msg = 'ERROR: Control_TMA_info.xlsx could not be read';
        logMsg(C,msg,1);
        return        
    end
    v = a.Properties.VariableNames;
    ncol = size(a,2);    
    aa = table2cell(a);
    pp=cellfun(@isempty,aa);
    k = 1;
    o = [];
    for i=1:ncol
        w =split(v{i},'_');
        ts = w{1};
        tm = w{2};
        w = aa(~pp(:,i),i);
        for j=1:numel(w)
            ww = split(w{j},',');
            tma(k) = str2num(tm);
            tissue{k} = ts;
            tx(k) = str2num(ww{1});
            ty(k) = str2num(ww{2});
            core{k} = sprintf('[1,%d,%d]',tx(k),ty(k));
            k=k+1;
        end
    end
    %    
    t = table(tma',tx',ty',core',tissue');
    t.Properties.VariableNames = {'TMA','cx','cy','Core','Tissue'};
    t = sortrows(t,[2,3]);
    t.ncore = (1:k-1)';
    t.project = 0*t.ncore+C.project;
    t.cohort  = 0*t.ncore+C.cohort;
    t = t(:,[end-2:end,1:end-3]);
    %
    if (C.opt==0)
        fname = fullfile(C.dbload,...
            sprintf('project%d_ctrlcores.csv',C.project));
        writetable(t,fname);
    end
    %
end