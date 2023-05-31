function R = vert2Poly(C)
%%-------------------------------------------------------------
%% convert the vertex arrays to WKT polygons in pixel coords
%% 2020-02-10   Alex Szalay
%%-------------------------------------------------------------
    %
    % go through each region first
    %
    R = C.PR;
    for i=1:numel(R.rid)
        %
        r = R(i,:);
        v = P.V(P.V.regionid==r.regionid,:);
%        x = C.pscale*[v.x;v.x(1)];
%        y = C.pscale*[v.y;v.y(1)];
        x = [v.x;v.x(1)];
        y = [v.y;v.y(1)];
        z = int32(floor([x';y']'));
        %
        if (r.isNeg)
            z = flipud(z);
        end
        %
        s{i} = ['POLYGON ((', char(join(join(string(z))',',')),'))'];
        %
    end
    %
    R.poly = s';
    %
end