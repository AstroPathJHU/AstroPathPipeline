function p=showPolygons(C,flag)
    %
    % show the layout first, with the QPTiff image
    %
    showLayout(C,flag);
    hold on
    %
    M = max(C.P.m);
    c = {'g','b'};
    w = C.P.x(C.P.m==0)+1;
    %
    for m=1:M
        %
        p  = C.P(C.P.m==m,:);
        tx = C.qpscale*[p.x; p.x(1)];
        ty = C.qpscale*[p.y; p.y(1)];
        %      
        plot(tx, ty,'-','Color',c{w(m)},'LineWidth',3);
        hold on
    end
    %
    hold off
    shg
    %
end

