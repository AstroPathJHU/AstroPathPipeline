function W = makeWarp(C,varargin)
%%-----------------------------------------------------------------------
%% Cross-register piecewise tiles of size sz from the stitched Astropath
%% and QPTIFF images, and return displacement map. The (x,y) coordinates 
%% correspond to the lower left corner of each tile, they need to be
%% offset by sz/2 for the center. The columns dx,dy are the displacements 
%% of the QP (O.q) image relative to the AP (O.b) in actual scaled ipix.
%%
%% Returns a table with columns n,x,y,dx,dy,mi
%%
%% 2020-05-19   Alex Szalay
%%-------------------------------------------
    %
    logMsg(C,'makeWarp started');
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %------------------------
    % get the bounding box
    %------------------------
    sz = C.sz;
    mx1 = min(C.H.mx1)/C.ppscale;
    mx2 = max(C.H.mx2)/C.ppscale;
    my1 = min(C.H.my1)/C.ppscale;
    my2 = max(C.H.my2)/C.ppscale;
    %
    mx2 = min(mx2,size(C.aimg,2)-sz);
    my2 = min(my2,size(C.aimg,1)-sz);
    %
    n1 = max(floor(my1/sz)-1,1);
    n2 = floor(my2/sz)+1;
    m1 = max(floor(mx1/sz)-1,1);
    m2 = floor(mx2/sz)+1;
    %
    w = [];
    k = 1;
    for i=n1:n2
        y = sz*(i-1);
        for j=m1:m2
            x = sz*(j-1);
            %fprintf('%d: %d %d %d %d\n',k,i,j,x,y);                
            Z = xcregister(C,sz,x,y);
            if (Z.flag==1)
                w(k,:) = [k,Z.x,Z.y,Z.dx,Z.dy,Z.mi];
                k = k+1;
            end
        end
    end
    %-----------------------
    % pack it into a table
    %-----------------------
    W = table(int32(w(:,1)),int32(w(:,2)),int32(w(:,3)),...
        w(:,4),w(:,5),w(:,6));
    W.Properties.VariableNames={'n','x','y','dx','dy','mi'};
    %--------------------------------------------------
    % find the connected regions and add them to W.gc
    %--------------------------------------------------
    W = findConnected(W);
    %-----------------------------------------
    % write it to a file to help debugging
    %-----------------------------------------
    if (opt>0)
        writetable(W,sprintf('%s-warp-%d.csv',C.samp,sz));
    end
    %
    logMsg(C,'makeWarp end');
    %
end

