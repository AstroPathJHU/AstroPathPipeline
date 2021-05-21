function R = vert2Poly(C)
%%-------------------------------------------------------------
%% convert the vertex arrays to WKT polygons in pixel coords
%% 2020-02-10   Alex Szalay
%%-------------------------------------------------------------
    %
    % go through each region first
    %
    R = C.PR;
    %
    for i=1:numel(R.rid)
        %
        r = R(i,:);
        v = C.PV(C.PV.regionid==r.regionid,:);
        x = [v.wx;v.wx(1)];
        y = [v.wy;v.wy(1)];
        z = int32(floor([x';y']'));
        %
        if (r.isNeg)
            z = flipud(z);
        end
        %
        nvert(i) = numel(x)-1;
        s{i} = ['POLYGON ((', char(join(join(string(z))',',')),'))'];
        %
    end
    %
    R.poly = s';
    R.nvert = nvert';
    %
end