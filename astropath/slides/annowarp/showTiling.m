function w = showTiling(W)
%%-------------------------------------------------
%% Show the contour map of the warp fields as well
%% as the surfaces of the warps as a function of X,Y
%%
%% 2020-05-18   Alex Szalay
%%-------------------------------------------------
    %
    figure(4);
    %
    n=2;
    m=2;
    k=1;
    gps = unique(W.gc);
    subplot(n,m,k);
        hold on
        for i=1:numel(gps)
            w = W(W.gc==gps(i),:);
            if (numel(w.n)>10)
                surfir(double(w.x),double(w.y),double(w.dx));
            end
        end    
        hold off
        title('dx')
        view(360,-90);
        axis equal
        box on
        k = k+1;
    subplot(n,m,k);
        hold on
        for i=1:numel(gps)
            w = W(W.gc==gps(i),:);
            if (numel(w.n)>10)
                surfir(double(w.x),double(w.y),double(w.dy));
            end
        end        
        hold off
        title('dy')
        view(360,-90);
        axis equal
        box on  
        k = k+1;
    subplot(n,m,k);
        hold on
        for i=1:numel(gps)
            w = W(W.gc==gps(i),:);
            if (numel(w.n)>10)
                surfir(double(w.x),double(w.y),double(w.dx));
            end
        end    
        hold off
        title('dx');
        view(0.3,-8.4);
        xlabel('x')
        box on
        k = k+1;
    subplot(n,m,k);
        hold on
        for i=1:numel(gps)
            w = W(W.gc==gps(i),:);
            if (numel(w.n)>10)
                surfir(double(w.x),double(w.y),double(w.dy));
            end
        end        
        hold off
        title('dy');
        view(90.15,1.49);
        xlabel('y');
        box on
        k = k+1;
    shg
    %
end

