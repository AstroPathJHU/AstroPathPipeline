function T=getBatchInfo(C)
%%-------------------------------------
%% collect all the Batch information,
%% add layer numers
%%
%% 2020-08-06
%%-------------------------------------
    %
    f = fullfile(C.batch,'Merge*.csv');
    d=dir(f);
    if (numel(d)==0)
        C.err=1;
        msg = 'ERROR: No MergeConfig.csv file found';
        logMsg(C,msg,1);
        return
    end
    %
    T=[];
    for i=1:numel(d)
        t = readtable(fullfile(d(i).folder, d(i).name));
        t.layer = (1:numel(t.BatchID))';
        T = [T;t];
    end
    %
    msg = sprintf('%d MergeConfig.csv files found',numel(d));
    logMsg(C,msg);    
    %{
    if (C.opt==0)
        fname = fullfile(C.dbload,...
            sprintf('project%d_batch.csv',C.project));
        writetable(T,fname);
    end
    %}
end