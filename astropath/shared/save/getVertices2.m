function [v,s] = getVertices(p,a,regionid)
%%---------------------------------------------
%% get all the vertices from a given polygon.
%% Positive are counterclockwise
%% Holes are counterclockwise
%% p: is the XML struct with the vertex points
%% a: is teh attributes for the region
%%
%% Alex Szalay, Baltimore, 2019-02-12
%%---------------------------------------------
    %
    v = [];
    rid = [];
    vid = [];
    x   = [];
    y   = [];
    z   = [];
    for k=1:numel(p)
        rid(k) = regionid;
        vid(k) = uint32(k);
        x(k)   = uint32(str2num(p{k}.Attributes.X));
        y(k)   = uint32(str2num(p{k}.Attributes.Y));
        z(k,:) = [x(k),y(k)];
    end
    %
    % close the curve
    %
    z = [z;[x(1),y(1)]];
    %
    % if hole, reverse the array
    %
    if (a.NegativeROA)
        z = flipud(z);
    end
    %
    va = table(rid',vid',x',y');
    va.Properties.VariableNames = {'regionid','vid','x','y'};
    v  = [v;va];
    %
    s = ['(', char(join(join(string(z))',',')),')'];
    %
end

