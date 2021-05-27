function d = runZoomAll()
%%--------------------------------------------------
%% Run the prepSample code on all samples
%% in the root1 directory
%%
%% Alex Szalay, Baltimore, 2019-02-13
%%--------------------------------------------------
global logctrl
    %
    logctrl=0;
    %
    d = readtable('\\bki02\c\BKI\save\samples.csv');
    n = numel(d.SampleID);
    %
    for i=1:n
        samp = d.SlideID{i};
        runZoom(samp,1);
    end
    %
end