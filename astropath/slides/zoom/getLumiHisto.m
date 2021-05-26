function h = getLumiHisto()
%%-----------------------------------------
%% get the mean image of the luminescence
%% images of a given slide
%%-----------------------------------------
    %
    ff = 'y:\lumi\histo\';
    if (exist(ff)==0)
        mkdir(ff);
    end
    %
    t = readtable('W:\bki\save\samples1.csv');
    for n=3:numel(t.SampleID)
        %
        samp = t.SlideID{n};
        %fprintf('%s\n',samp);
        h = getHistogram(samp);
        g = ['Y:\lumi\histo\',samp,'-lumi.csv'];
        fprintf('%s\n',g);
        try
            writetable(h,g);
        catch
            return
        end
        %
    end
    %
end


function t = getHistogram(samp)
%%----------------------------------------------------
%% build the stacked histogram of the log fluxes
%% so that we can compare the different slides
%%----------------------------------------------------
    %
    ff = ['Y:\lumi\',samp,'\*.tif'];
    d  = dir(ff);
    %
    b = linspace(-2,6,401);
    h = zeros(1,400);
    for n=1:numel(d)
        a = imread([d(n).folder,'\',d(n).name]);
        ix = a(:)>0;
        h = h + histcounts(log(a(ix)),b);
    end
    h = h/numel(d);
    %
    t = table(b(2:end)',h');
    t.Properties.VariableNames={'b','h'};
    %
end
