function T = getCells(C,n,flag)
%%-------------------------------------------------
%% compute the outlies of cell membranes and nuclei
%%
%% Alex Szalay, 2019-03-09
%%-------------------------------------------------
    %
    t0 = cellBoundary(C,n,0);
    t1 = cellBoundary(C,n,1);
    t2 = cellBoundary(C,n,2);
    t3 = cellBoundary(C,n,3);
    if (C.err==1)
        return
    end
    %
    T = [t0;t1;t2;t3];
    %
    if (~isempty(T) & flag ==0)
        f = fullfile(C.geom,[replace(C.H.file{n},...
            '.im3','_cellGeomLoad.csv')]);
        writetable(T,f);
    end
    %
end