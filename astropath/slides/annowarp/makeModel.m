function M = makeModel(C)
%%------------------------------------------------------
%% Build the columns for the linear model
%% W: table of input displacement field
%% M: output table (copy of W) with added columns
%%   cx,cy: the centers of each sz-size patch
%%   n,m: the indexes of the big tiles (n:x, m:y).
%%   GX,GY: the corners of the big tiles
%%   xx,yy: relative patch centers within a big tile
%%
%% 2020-05-20   Alex Szalay
%%------------------------------------------------------
    %
    M = C.Y.W;
    E = C.Y.E;
    M.Properties.VariableNames{1}='i';    
    %----------------------------------------------
    % create the centers (cx,cy) for the sz patches
    %----------------------------------------------
    M.cx = double(M.x+C.sz/2);
    M.cy = double(M.y+C.sz/2);
    %------------------------------------------------
    % determine which (n,m) tile contains each point
    %------------------------------------------------
    %tx = min(diff(E.ex));
    %ty = min(diff(E.ey));
    %
    M.n = floor(double(M.cx)/E.tx);
    M.m = floor(double(M.cy)/E.ty);
    %--------------------------------------------------------
    % build the coordinates relative to the big tile corners
    %--------------------------------------------------------
    M.gx = M.n*E.tx;
    M.gy = M.m*E.ty;
    %
    M.xx = M.cx-M.gx;
    M.yy = M.cy-M.gy;
    M.e  = 1+0*M.x;
    %
end
