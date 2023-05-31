function [R,V] = getRegions(a,n)
%%-----------------------------------------
%% get the Regions, pack them into a table
%%
%% Alex Szalay, Baltimore, 2019-02-12
%%-----------------------------------------
    %    
    R    = [];
    V    = [];
    %
    b = a.Regions.Region;
    M = numel(b);
    for m=1:M
        %
        if (M==1)
            p = b.Attributes;
            v = b.Vertices.V;
        else
            p = b{m}.Attributes;
            v = b{m}.Vertices.V;
        end
        %
        rid = uint32(m+1000*n);
        nv  = uint32(numel(v));
        %
        [vt,s] = getVertices(v,p,rid);
        V   = [V;vt];
        %
        poly = string(['POLYGON ', s, ')']);
        smp = 0;
        ra = table(rid,0,n,m,str2num(p.NegativeROA),...
            string(p.Type),nv,poly);
        ra.Properties.VariableNames = {'regionid','sampleid','layer',...
            'rid','isNeg','type','nvert','poly'};
        R  = [R;ra];        
    end
    %
end
