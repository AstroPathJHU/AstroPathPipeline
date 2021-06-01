function E = getEdges(C)
%%--------------------------------------------
%% Create the grid layout over the images
%%
%% 2020-07-10   Alex Szalay
%%-----------------------------------------
    %
    logMsg(C,'getEdges');
    %
    DELTAX = 1400;
    DELTAY = 2100;
    %--------------------------
    % get the bounding box
    %--------------------------
    mx1 = min(C.H.mx1)/C.ppscale;
    mx2 = max(C.H.mx2)/C.ppscale;
    my1 = min(C.H.my1)/C.ppscale;
    my2 = max(C.H.my2)/C.ppscale;
    %
    E.nx1 = max(1,floor(mx1/DELTAX));
    E.nx2 = floor(mx2/DELTAX)+1;
    E.ny1 = max(1,floor(my1/DELTAY));
    E.ny2 = floor(my2/DELTAY)+1;
    %
    E.ex = DELTAX*(E.nx1:E.nx2);
    E.ey = DELTAY*(E.ny1:E.ny2);
    %
    E.tx = DELTAX;
    E.ty = DELTAY;    
    %
end