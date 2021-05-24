function d = geomCellsAll(root1)
%%--------------------------------------------------
%% Run the prepSample code on all samples
%% in the root1 directory
%%
%% Alex Szalay, Baltimore, 2019-02-13
%%--------------------------------------------------
global logctrl
    %
    logctrl=1;
    %
    root1 = '\\bki03\Clinical_Specimen_BMS_03';    
    d = readtable('..\..\save\BMS_03.csv');
    n = numel(d.SampleID);    
    %
    for i=1:n
        samp = d.SlideID{i};
        geomCells(root1, samp);
    end
    %
end