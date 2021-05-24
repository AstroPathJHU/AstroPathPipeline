function [W] = findConnected(W)
%%-----------------------------------------------------
%% Find the connected pieces of the warp map
%%
%% 2020-07-04   Alex Szalay
%%-----------------------------------------------------
    %
    ux = unique(W.x);
    uy = unique(W.y);
    dx = ux(2)-ux(1);
    dy = uy(2)-uy(1);
    %
    % create affinity matrix
    %
    N = numel(W.n);
    A = sparse(zeros(N,N));
    %
    % the horizontal links first
    %
    for i=1:numel(ux)-1
        ix1 = W.x==ux(i);
        ix2 = W.x==ux(i+1);
        wy  = intersect(W.y(ix1),W.y(ix2));
        iy  = ismember(W.y,wy);    
        %
        iw = find(ix1 & iy);
        jw = find(ix2 & iy);
        %
        for k=1:numel(iw)
            A(iw(k),jw(k))=1;
            A(jw(k),iw(k))=1;
        end
    end
    %
    for i=1:numel(uy)-1
        iy1 = W.y==uy(i);
        iy2 = W.y==uy(i+1);
        wx  = intersect(W.x(iy1),W.x(iy2));
        ix  = ismember(W.x,wx);    
        %
        iw = find(iy1 & ix);
        jw = find(iy2 & ix);
        %
        for k=1:numel(iw)
            A(iw(k),jw(k))=1;
            A(jw(k),iw(k))=1;
        end
    end
    %
    a = sparse(A);
    [npart,gc] = graphconncomp(A,'Directed',false);
    %c = grpstats(gc,gc,'numel');
    %gps = find(c>20)
    %
    W.gc =gc';
    %
end