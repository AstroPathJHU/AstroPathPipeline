function o = wkt2path(w)
%%-------------------------------------------------------
%% convert polygons from wkt format into clipper paths
%%-------------------------------------------------------
    %
    for i=1:numel(w)
        s = w{i};
        s = replace(s,'POLYGON ((','');
        s = replace(s,')','');
        s = split(s,',');
        c =[];
        for j=1:numel(s)
            c(j,:) = cellfun(@str2num,split(s{j}))';
        end
        o(i).x = int32(c(1:end-1,1));
        o(i).y = int32(c(1:end-1,2));
    end
    %  
end