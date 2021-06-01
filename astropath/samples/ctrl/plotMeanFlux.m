function C=plotMeanFlux(varargin)
%%----------------------------------------
%% plot the raw data
%%----------------------------------------
    %
    opt = 1;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    if (ishandle(opt))
        close(opt);
    end
    %
    h1=figure(opt);
    set(h1,'Position',[200,300,1800,900]);
    %
    C = getPmts('Y:\ctrl13\csv');
    C.samp = 'meanFlux';
    C.D = readtable([C.csv,'\meanflux.csv']);
    C.cal = table2array(readtable([C.csv,'\calibrations.csv']));
    C.batch = unique(C.D.batch);
    %
    annotation('textbox',[0,0.95,1,0.05],...
       'String',C.samp,'EdgeColor','none',...
       'HorizontalAlignment','center','FontSize',12);
    %
    P.axis = [min(C.batch)-1 max(C.batch)+1 -3 3];
    for m=2:7
        subplot(2,3,m-1);
            hold on;
            plotOneMarker(C,m,P);
    end
    shg
    %
end


function p = plotOneMarker(C,m,P)
    %
    D=C.D;
    marks   = {'o','^','s','<','>','d'};
    color   = colormap(lines);
    batch   = unique(D.batch);
    %
    title(C.markers.marker{m});
    %
    for t=1:numel(C.tissues.ntissue)
        %cores  = C.cores(C.cores.ntissue==t,:);
        %for i=1:numel(cores.core)
        %    c = cores.core(i);
            ix = D.nmarker==m & D.ntissue==t;
            d  = D(ix,:);
            d  = sortrows(d,'batch');
            %
            if (numel(d)==0)
                continue
            end
            %
            clr = color(t,:);
            p(t) = plot(d.batch,log(d.m0),'-',...
                'MarkerSize',5, 'MarkerFaceColor',clr,...
                'MarkerEdgeColor',clr*0.4,...
                'Marker',marks{t},'Color',clr);
        %end
    end
    %
    t = t+1;
    p(t) = plot(C.cal(:,1),log(C.cal(:,m)),'-',...
                'LineWidth',1.5, 'Color',[0.2,0.4,0.8]);
    
    axis(P.axis);
    box on
    grid on
    xlabel('batch');
    ylabel('log(flux)');
    xticks((min(batch)-1:max(batch)+1));
    lgd = {C.tissues.tissue{1},C.tissues.tissue{2},'Calibration'};
    legend(p,lgd);
    %
end
