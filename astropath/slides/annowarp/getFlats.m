function F = getFlats(C)
%%-------------------------------------------------
%% Get the mean of the residuals for each tile.
%% Returns a table with the mean value of the displacements
%% n,m:     index of the tile
%% tx,ty:   the constant offset in the displacements
%%
%% 2020-05-21   Alex Szalay
%%-------------------------------------------------
    %
    logMsg(C,'getFlats');
    %--------------------------------------------
    % create a list of all the n,m in the data
    % offset by 1 for MATLAB indexing
    %--------------------------------------------
    M = C.Y.M;
    subs = table2array(M(:,{'n','m'}))+1;
    %
    % compute the mean of the residuals over all points in a tile
    %
    rmeany = accumarray(subs,M.ry,[],@mean);
    rmeanx = accumarray(subs,M.rx,[],@mean);    
    %
    % convert the matrix into a linear table form
    % with the subscripts maped onto n,m starting at 0
    %
    [nx,ny] = size(rmeanx);
    [ix,iy] = ind2sub(size(rmeanx),(1:nx*ny));
    %   
    F = table(ix'-1,iy'-1,rmeanx(:),rmeany(:));
    F.Properties.VariableNames={'n','m','fdx','fdy'};
    %
end