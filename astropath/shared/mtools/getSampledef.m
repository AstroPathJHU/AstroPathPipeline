function d = getSampledef(C)
%%-----------------------------------------
%% return the samples as defined by the 
%% sampledef.csv file
%%
%% 2020-06-18   Alex Szalay
%%-----------------------------------------
    %
    d=[];
    sampledef = fullfile(C.root,'sampledef.csv');
    if (exist(sampledef)~=2)
        msg = sprintf('ERROR: cannot find %s',sampledef);
        fprintf('0,0,%s,%s\n',C.samp,msg);
        return
    else
        d = readtable(sampledef);    
    end
    %