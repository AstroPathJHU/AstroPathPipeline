function C = runAnnowarp(root,samp,varargin)
%%--------------------------------------------------------------
%% Run the steps to create the warp map for the annotations
%% opt>0 means do not overwrite the <samp>-vertices.csv file
%%
%% 2020-05-21   Alex Szalay
%%--------------------------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    C = getConfig(root,samp,'annowarp');
	logMsg(C,'runAnnowarp started',1);
    %
    warning('off',  'stats:regress:RankDefDesignMat');
    warning('off','curvefit:fit:iterationLimitReached');
	%
    C = readMetadata(C);
    if (C.err==1)
        return
    end
    %
    C.sz = 100;
    %----------------    
    % get the images
    %----------------
    C = getImages(C);
    if (C.err==1)
        return
    end
    %---------------------------
    % calculate the grid edges
    %---------------------------
    C.Y.E = getEdges(C);
    %
    C.qshifty=0;
%    if (strcmp(C.samp,'M211')>0)
    %-----------------------------------
    % tweak the y position by -900 for
    % the microsocope glitches
    %-----------------------------------
    if (C.yposition==0)
        C.qshifty = 900;
    end
    %
    if (opt==-1)
        return
    end    
    %    
    try
        C.Y.W  = makeWarp(C);
        %C.Y.WW = C.Y.W;
    catch
        logMsg(C,'ERROR: in makeWarp',1);
        return
    end
    if (opt==-2)
        return
    end
    %----------------------
    % clean the outliers
    %----------------------
    try
        C.Y.W = cleanWarp(C);
    catch
        logMsg(C,'ERROR: in cleanWarp',1);
        return
    end
    if (opt==-3)
        return
    end
    %---------------------
    % build the models
    %---------------------
    C = createModel(C);    
    if (opt==-4)
        return
    end
    %--------------------------
    % build interpolation map
    %--------------------------
    C.Y.T = createMap(C);
    if (opt==-5)
        return
    end    
    %---------------------------------------------
    % Update vertices and polygons, write to disk
    %---------------------------------------------
    C.PV = warpAnnotations(C);
    C.PR = vert2Poly(C);
    %
    if (opt==0)
        fname = fullfile(C.dbload,[C.samp,'_vertices.csv']);
        writetable(C.PV,fname);
        %
        fname = fullfile(C.dbload,[C.samp,'_regions.csv']);
        writetable(C.PR,fname);
    end
    %
    logMsg(C,'annowarp finished',1);
    %
return



