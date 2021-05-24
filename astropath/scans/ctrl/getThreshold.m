function thr = getThreshold(b)
%%-----------------------------------------------------
%% determine the threshold for the tissue separation
%% from the log histogram of the image
%%
%% 2020-08-07   Alex Szalay
%%-----------------------------------------------------
    %
    xmax = asinh(max(b(:)));
    xmin = 1.0;
    t  = 0.10;
    %
    x  = linspace(xmin,xmax,400);
    y  = histc(asinh(b(:)),x)';
    yc = cumsum(y)/sum(y);
    thr = x(find(yc>t,1));
    %
end


