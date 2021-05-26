function showResults(C)
    %
    if (C.flag==1)
        m = 4;
        figure(1);
        %set(gcf,'Position',[100,100,1200,800]);
        %
        subplot(2,3,3);
            plot(C.G.dx1,C.G.dy1,'.','MarkerSize',m);
            axis([-10 10 -10 10]);
            axis square
        subplot(2,3,6);
            plot(C.G.dx2,C.G.dy2,'r.','MarkerSize',m);
            axis([-2 2 -2 2]);
            axis([-10 10 -10 10]);
            axis square
        %        
        subplot(2,3,[4,5]);
            C = showLayout(C,2);
        %
        subplot(2,3,[1,2]);
            showAffine(C);
        %
    end
    %
    if (C.flag==2)
        %
        m = 4;
        figure(1);
        set(gcf,'Position',[100,100,1200,600]);
         annotation('textbox',[0,0.84,1,0.1],...
            'String',C.samp,...
            'EdgeColor','none',...
            'HorizontalAlignment','center',...
            'FontSize',12,'Interpreter','none');
        %
        subplot(1,2,1);
            plot(C.G.dx,C.G.dy,'.','MarkerSize',m);
            axis([-8 8 -8 8]);
            axis square
        subplot(1,2,2);
            plot(C.G.dzx,C.G.dzy,'r.','MarkerSize',m);
            axis([-8 8 -8 8]);
            axis square
        %
    end
    %
    shg
    %
end
