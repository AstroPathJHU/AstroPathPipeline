function R = fitModel(C)
%%------------------------------------
%% Perform the model fit
%%
%% 2020-07-11   Alex Szalay
%%------------------------------------
    %
    logMsg(C,'fitModel');
    %--------------------
    % xx, yy, e, n, m
    % 10  11  12 6  7
    %--------------------
    M = C.Y.M;
    E = C.Y.E;
    V = double(table2array(M(:,{'xx','yy','e','n','m'})));
    if (rank(V)<size(V,2))
        msg = sprintf('WARNING: rank of fitModel matrix is %d',...
            rank(V));
        logMsg(C,msg,1);
    end
    %
    R.bx = regress(M.dx,V);
    R.by = regress(M.dy,V);
    %----------------------------
    % also add edge definitons
    %----------------------------
    R.tx = E.tx;
    R.ty = E.ty;
    %
end