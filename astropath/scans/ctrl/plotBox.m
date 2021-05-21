function p = plotBox(C,mode,varargin)
    %
    opt = 1;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    if (ishandle(opt))
        close(opt);
    end
    h1=figure(opt);
    set(h1,'Position',[200,300,1000,400]);    
    %
    color   = colormap(lines);
    %
    d = readtable(fullfile(C.csv,'flatflux.csv'));
    f=d.f0;
    clr = color(d.ntissue,:);
    batch = unique(d.batch);
    w = 0.4*ones(1,numel(d.batch));
    if (mode==1)
        boxplot(f,[d.batch,d.ntissue], 'BoxStyle','outline',...
            'PlotStyle','traditional','ColorGroup',d.ntissue,...
            'Widths',w,'FactorGap',[1,1],'FactorSeparator',[1]);
        xlabel('batch/tissue');
    else
        boxplot(f,d.batch, 'BoxStyle','outline'); %,...
        %    'PlotStyle','traditional','ColorGroup',d.ntissue,...
        %    'Widths',w,'FactorGap',[1,1],'FactorSeparator',[1]);
        xlabel('batch');
    end
    %
    set(findobj(gca,'type','line'),'linew',1)
    box on
    grid on
    ylabel('calibrated flux');
    ylim([0,2]);
    title('Calibration vs Batch');
    shg    
    %
end