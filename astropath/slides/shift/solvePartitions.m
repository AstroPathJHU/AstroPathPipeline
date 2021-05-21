function [C,A] = solvePartitions(C)
%%------------------------------------------------------
%% Create the partitions of the adjacency matrix,
%% as well as the weighted edges for the displacements.
%%
%% Alex Szalay, Baltimore, 2018-11-12
%%------------------------------------------------------
    %
    logMsg(C,mfilename);
    %
    C.limit = 8;
    %
    % symmetrize the shifts, ignore shifts from blank overlaps
    %
    q = symShifts(C);
    %
    % use the 4-connected edges and the overlaps with normal shifts
    %    
    ix = q.p1<q.p2;
    ix = ix & ismember(q.code,[2,4,6,8]);
    %ix = ix & abs(q.dx)<C.limit & abs(q.dy)<C.limit;
    %
    q = q(ix,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % find the nonconnected members at the end
    % and add them back to the array
    %
    nc = find(~ismember(C.H.n,q.p1));
    ic = find(nc>max(q.p1));
    nc = nc(find(nc>max(q.p1)));
    mc = max(q.n);
    for i=numel(nc)
        v = {mc+i, 0, nc(i), nc(i), 0,0,1,0};
        q = [q;v];
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    h1 =  [q.p1;q.p2];
    h2 =  [q.p2;q.p1];
    dx = -[q.dx;-q.dx];
    dy = -[q.dy;-q.dy];
    %
    C.q = q;
    %
    %  get displacements and adjacency matrix
    tx = sparse(h1,h2,dx);    
    ty = sparse(h1,h2,dy);
    A  = sparse(h1,h2,1);
    %
    % determine number of partitions, and their cardinalities
    %
    [npart,gc] = graphconncomp(A,'Directed',false);
    c = grpstats(gc,gc,'numel');
    %
    for i=1:numel(c)
        logMsg(C,sprintf('Created partition(%d).nfields=%d',i,c(i)));
    end
    %
    C.A = A;
    C.tx = tx;
    C.ty = ty;
    C.gc = gc;
    %return
    %
    %{
    % if there were unattached fields at the end of the fields
    % that were left off of A, we should add them back to gc
    %
    nc = numel(C.H.n) - size(A,1);
    %    
    for i=1:nc
        gc = [gc,0];
        c  = [c;1];
    end
    %
    % set isolated partitions to zero and renumber the others
    %
    ic =(c<3);
    c = c(~ic);
    ix = find(~ic);
    iy = ~ismember(gc,ix);
    gc(iy)=0;    
    %
    for i=1:numel(ix)
        gc(gc==ix(i))=i;
    end
    npart = numel(ix);
    %}    
    % add the partition groups to C.H
    %
    C.H.gc = gc';
    %
    % save all the partitions
    %
    for n=1:npart
        %
        g = find(C.H.gc==n);
        W{n}.g = g';
        W{n}.ng = numel(g);
        W{n}.X = sum(tx(g,g));
        W{n}.Y = sum(ty(g,g));
        W{n}.A  = A(g,g);
        %
    end
    %
    C.W = W;
    %
end

