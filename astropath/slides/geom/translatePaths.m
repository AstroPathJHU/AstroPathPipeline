function q = translatePaths(q,dx,dy)
%%-----------------------------------------------------
%% take a clipper Paths object and translate by dx,dy
%%
%% Alex Szalay, Baltimore, 2019-03-03
%%-----------------------------------------------------
    %
    px = int32(floor(dx));
    py = int32(floor(dy));
    %
    if (numel(q)>0)
        for k=1:numel(q)
            q(k).x = q(k).x + px;
            q(k).y = q(k).y + py;
        end            
    end
    %
end