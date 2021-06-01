function [T,F] = getBoundaries(C)
%%-----------------------------------------------------------
%% Extract a given layer from the component TIFF images
%%-----------------------------------------------------------
    %
    logMsg(C,'getBoundaries');
    %
    E = int32(0*C.H.n+1);
    %
    T=[];
    for i=1:numel(C.H.n)
        %
        h = C.H(i,:);
        P{i} = rect2Path(h.mx1,h.my1,h.mx2,h.my2);
        R{i} = ['POLYGON (',path2char(P{i}),')'];
        %
        [t,q{i}] = getTumorLayer(C,i);
        if (numel(t)>0)
            T = [T;t];
        end
        %
    end
    F = table(C.H.n,E,R');
    F.Properties.VariableNames = {'n','k','poly'};
    %
end


function r = rect2Path(x1,y1,x2,y2)
%%--------------------------------------------
%% create a rectangle polygon in clipper format
%%
%% Alex Szalay, Baltimore, 2019-03-03
%%--------------------------------------------
    %
    r.x = int32(floor([x1,x2,x2,x1]))';
    r.y = int32(floor([y1,y1,y2,y2]))';
    %
end

