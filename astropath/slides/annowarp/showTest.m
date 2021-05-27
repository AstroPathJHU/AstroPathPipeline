function showTest(T)
    %
    figure(10);
    subplot(2,2,1);
        plot(T.px10','.');
        hold on;
        plot(T.py01,'.');
        hold off
        title('diagonal');
    subplot(2,2,2);
        plot(T.px01','.');
        hold on;
        plot(T.py10,'.');
        hold off
        title('off-diagonal');
    subplot(2,2,3);
        plot(T.px00','.');
        hold on;
        plot(T.py00,'.');
        hold off
        title('offsets');
    subplot(2,2,4);
        plot(T.px00',T.px10,'.');
        hold on;
        plot(T.py00,T.py01,'.');
        hold off
        title('xplot');
    shg
    %
end