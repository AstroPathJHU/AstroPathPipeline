function d = geomSampleAll()
%%--------------------------------------------------
%% Run the same command on all samples
%% in the root1 directory
%%
%% Alex Szalay, Baltimore, 2019-02-13
%%--------------------------------------------------
global logctrl
    %
    logctrl = 1;
    %
    root1 = '\\bki03\Clinical_Specimen_BMS_03';    
    d = readtable('..\..\save\BMS_03.csv');
    n = numel(d.SampleID);    
    %
    for i=1:n
        samp = d.SlideID{i};        
        fprintf('%s\n',samp);
        geomSample(root1,samp);
    end
    %
    logctrl = 0;
    %
end