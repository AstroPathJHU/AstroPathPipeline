function d = runZoomAll()
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
    root1 = '\\bki02\e\Clinical_Specimen';
    d = readtable('\\bki02\c\BKI\save\samplesX.csv');
    n = numel(d.SampleID);
    %
    for i=1:n
        samp = d.SlideID{i};
        runZoom(root1, samp);
    end
    %
end