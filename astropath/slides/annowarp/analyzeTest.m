function analyzeTest(T)
    MPX00 = mean(T.px00);
    SPX00 = std( T.px00);
    MPY00 = mean(T.py00);
    SPY00 = std( T.py00);
    
    [MPX00,SPX00,MPY00,SPY00]
    ix = abs(T.px00-MPX00)<SPX00 & abs(T.py00-MPY00)<SPY00; 
    t = T(ix,:);
    [numel(T.px00),numel(t.px00)]
    %
    mpx00 = mean(t.px00);
    mpy00 = mean(t.py00);
    mpx10 = mean(t.px10);
    mpx01 = mean(t.px01);
    mpy10 = mean(t.py10);
    mpy01 = mean(t.py01);

    [mpx00,mpx10,mpx01,mpy00,mpy10,mpy01]
    
    spx00 = std(t.px00);
    spy00 = std(t.py00);

    
    %
    % now write the 3 sigma outliers
    %
    sigma = 2.5;
    ix = abs(T.px00-mpx00)>sigma*spx00 & abs(T.py00-mpy00)>sigma*spy00; 
    tt = T(ix,:);
    tt.samp
    
end