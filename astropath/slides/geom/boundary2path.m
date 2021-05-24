function P = boundary2path(B)
%%----------------------------------------
%% convert the boundary tracing polygon
%% to clipper Paths
%%----------------------------------------
    %
    P = [];
    if (numel(B)==0)
        return
    end
    %
    for k=1:length(B)
        b = B{k};
        P(k).x = int32(b(:,2));
        P(k).y = int32(b(:,1));
    end
    %
end