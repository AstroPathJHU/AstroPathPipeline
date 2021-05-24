function D = mergeLineSegments(C,varargin)
%%-----------------------------------------------------------------
%% Takes a list of regionid's and merges them into the first one.
%% Rewrites the regionids and the vertex lists as well as the 
%% polygon definitons
%%
%% Alex Szalay, 2020-11-17
%%-----------------------------------------------------------------
    %
    rlist = [1042,1044,1045,1046,1047,1048];
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %--------------------
    % make a copy of C
    %--------------------
    D = C;
    %--------------------------------------------------
    % make a copy of the regions and vertices in play
    %--------------------------------------------------
    O.PR = D.PR(ismember(D.PR.regionid,rlist),:);
    O.PV = D.PV(ismember(D.PV.regionid,rlist),:);
    %---------------------------------------------
    % delete the the redundant regions from D.PR
    % and all the involved vertices from D.PV
    %---------------------------------------------
    D.PR(ismember(D.PR.regionid,rlist(2:end)),:)=[];    
    D.PV(ismember(D.PV.regionid,rlist),:)=[];
    %---------------------------------------------
    % we need to do additional vertex editing
    % V will contain the modified vertices
    %---------------------------------------------
    V=[];
    ranges = {1:294,1:2306,1:2,1:252,3:203,1:3591};
    for i=1:numel(O.PR.regionid)
        regionid = O.PR.regionid(i);
        v = O.PV(O.PV.regionid==regionid,:);
        %---------------------------------
        % patch vertices in region 1045
        %---------------------------------
        if (regionid==1045)
            v = v(ranges{i},:);
            v.wx(1) = 2*29850;
            v.wy(1) = 2*9890;
            v.wx(2) = 2*29830;
            v.wy(2) = 2*9929;
        end
        V = [V;v(ranges{i},:)];        
    end
    %-------------------------------
    % now insert this back into O.PV
    %-------------------------------
    O.PV = V;
    %-------------------------------------------------
    % update the regionid of the vertices in O.PV
    % and recompute the polygon definition
    %-------------------------------------------------
    O.PV.regionid = O.PR.regionid(1)+0*O.PV.regionid;
    O.PV.vid = (1:numel(O.PV.vid))';    
    %------------------------------------
    % only keep the first region in O.PR
    % and recompute the polygons
    %------------------------------------    
    O.PR = O.PR(1,:);
    O.PR = vert2Poly(O);
    %----------------------------------------------------------
    % update the good region in D.PR and the vertices in D.PV
    %----------------------------------------------------------
    D.PR(D.PR.regionid==O.PR.regionid(1),:) = O.PR;    
    D.PV = [D.PV;O.PV];
    %---------------------------------------------
    % Update vertices and polygons, write to disk
    %---------------------------------------------    
    if (opt==0)
        fname = fullfile(C.dbload,[C.samp,'_vertices.csv']);
        writetable(D.PV,fname);
        %
        fname = fullfile(C.dbload,[C.samp,'_regions.csv']);
        writetable(D.PR,fname);
    end    
    %
end