function Z = cleanWarp(C)
%%----------------------------------------------
%% Clean the obvious outliers from the warp map
%% return the updated warp map
%%
%% 2020-07-10   Alex Szalay
%%----------------------------------------------
    %
    logMsg(C,'cleanWarp');

    %-----------------------------
    % handle each group separately
    %------------------------------
    W = C.Y.W;
    ix = [];
    gps = unique(W.gc);
    for i=1:numel(gps)
        gc = gps(i);
        w = W(W.gc==gc,:);
        %fprintf('%d: %d, %d\n',i, gc, numel(w.n));
        if (numel(w.n)>7)
            x = double(w.x);
            y = double(w.y);
            % test 
            ux = unique(x);
            uy = unique(y);
            if (numel(ux)==1 | numel(uy)==1)
                continue
            end
            fx = cFit1(x,y,w.dx);
            fy = cFit1(x,y,w.dy);
            dx = w.dx-feval(fx,[x,y]);
            dy = w.dy-feval(fy,[x,y]);
            ix = [ix, w.n(abs(dx)>10 | abs(dy)>10 )'];
        else
            % 
            ix = [ix,w.n'];
        end
    end
    %
    Z = W(~ismember(W.n,ix),:);
    %
end

%{
function [fitres] = cFit2(x, y, z)
    %
    [xx,yy,zz] = prepareSurfaceData( x, y, z );
    opts = fitoptions( 'Method', 'LinearLeastSquares','Robust','on');
    [fitres,gof] = fit([xx,yy],zz,fittype('poly11'), opts);
    %
end

%}



