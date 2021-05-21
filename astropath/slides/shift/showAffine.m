function q = showAffine(C)
%%----------------------------------------------------------------
%% show the residual displacements after correcting for an affine transformation
%%
%% Alex Szalay, Baltimore, 2018-10-07
%%----------------------------------------------------------------
    %
    logMsg(C,'showAffine');
    %
    x = C.H.x;
    y = C.H.y;
    g = find(C.H.gc>0); 
    q = quiver((x(g)-C.xposition),(y(g)-C.yposition),...
        C.V.ZX(g)-C.V.ax(g),C.V.ZY(g)-C.V.ay(g),...
        'Color',[0.25,0.25,0.65]);
    %
    hold on
    axis equal
    %
    if (isfield(C,'axis'))
        axis(C.axis);
    else
        axis([0 4.0 0 2.5]*1E4);
    end
    %
    c = 'brgmcybrgmcybrgmcybrgmcy';
    for n=1:numel(C.W)
        if(C.W{n}.skip==0)       
            in = (C.H.gc==n);
            p(n)=plot((C.H.x(in)-C.xposition),...
                (C.H.y(in)-C.yposition),...
                sprintf('.%s',c(n)),'MarkerSize',8);
        end
    end
    %
    xl = xlim;
    yl = ylim;
    tx = xl(2)*0.825;
    ty = yl(2)*0.9;
    text(tx,ty,C.samp,'FontSize',14,'Interpreter','none');
    %
    box on
    hold off
    shg
    %
end
