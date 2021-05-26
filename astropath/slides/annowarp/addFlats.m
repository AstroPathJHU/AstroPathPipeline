function M = addFlats(C);
%%------------------------------------------------------------
%% Add the flats values fdx, fdy to the M table, and 
%% update the model displacements mdx,mdy and residuals rx,ry.
%% 
%% 2020-05-21   Alex Szalay
%%------------------------------------------------------------
    %
    logMsg(C,'addFlats');
    %
    M = join(C.Y.M,C.Y.F);
    %
    M.mdx = M.mdx+M.fdx;
    M.mdy = M.mdy+M.fdy;
    M.rx  = M.rx -M.fdx;
    M.ry  = M.ry -M.fdy;    
    %
end