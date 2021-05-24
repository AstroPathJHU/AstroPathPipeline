function C = solveLaplace(C,n)
%%-------------------------------------------------------------
%% Solve the Laplacian with a pivot point for each partition
%% Alex Szalay, Baltimore, 2018-11-12
%%-------------------------------------------------------------
    %
    logMsg(C,mfilename);
    %
    % compute the Laplacian matrix
    %
    L = diag(sum(C.W{n}.A))-C.W{n}.A;
    C.W{n}.A=[];
    %    
    % pick pivot to be the last  of the partition
    %
    C.W{n}.pivot = C.W{n}.ng;
    %
    % create a logical array for selecting the non-pivot elements
    %
    ii = (C.W{n}.g>0);
    ii(C.W{n}.pivot) = 0;
    %
    % pin down the pivot element and get the displacement vectors
    %
    LL = L(ii,ii);
    TX = C.W{n}.X(ii);
    TY = C.W{n}.Y(ii);
    C.W{n}.X = [];
    C.W{n}.Y = [];
    %
    % convert to full matrix, do the pseudo-inverse
    %
    LL = full(LL);
    LI = pinv(LL);
    %
    % get the solution
    %
    x  = full(TX*LI);
    y  = full(TY*LI);
    %
    % reinsert zero for pivot element
    %
    x(C.W{n}.pivot) = 0;
    y(C.W{n}.pivot) = 0;
    %
    % create the position table
    %    
    Z = table(C.W{n}.g',x',y');
    Z.Properties.VariableNames = {'h','X','Y'};
    C.Z{n} = Z;
    %
end
