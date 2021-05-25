function plotRawFlux()
%%----------------------------------------
%% plot the raw data
%%----------------------------------------
    %
    C = getPmts('Y:\ctrl13\csv');
    C.samp = 'rawFlux';
    C.D = readtable([C.csv,'\',C.samp,'.csv']);    
    %
    annotation('textbox',[0,0.95,1,0.05],...
       'String',C.samp,'EdgeColor','none',...
       'HorizontalAlignment','center','FontSize',12);
    %
    P.axis = [15 24 -5 6];
    for m=2:7
        subplot(2,3,m-1);
            hold on;
            plotOneMarker(C,m,P);
    end
    shg
    %
end


function plotOneMarker(C,m,P)
    %
    D=C.D;
    marks   = {'o','^','s','<','>','d'};
    color   = colormap(lines);
    batch   = unique(D.batch);
    %
    title(C.markers.marker{m});
    %
    for t=1:numel(C.tissues.ntissue)
        cores  = C.cores(C.cores.ntissue==t,:);
        for i=1:numel(cores.core)
            c = cores.core(i);
            ix = D.nmarker==m & D.core==c;
            d  = D(ix,:);
            d  = sortrows(d,'batch');
            %
            if (numel(d)==0)
                continue
            end
            %
            clr = color(t,:);
            p(t) = plot(d.batch,log(d.f0),'-',...
                'MarkerSize',5, 'MarkerFaceColor',clr,...
                'MarkerEdgeColor',clr*0.4,...
                'Marker',marks{t},'Color',clr);
        end
    end
    %
    axis(P.axis);
    box on
    grid on
    xlabel('batch');
    ylabel('log(flux)');
    xticks((min(batch)-1:max(batch)+1));
    legend(p, C.tissues.tissue);
    %
end
