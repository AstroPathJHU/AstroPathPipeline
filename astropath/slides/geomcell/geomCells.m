function C = geomCells(root,samp,varargin)
%%---------------------------------------------------------
%% extract the relevant geometries from the InForm layers
%% and convert them into WKT polygon format
%%
%% Alex Szalay, Baltimore, 2019-03-07
%%---------------------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    C = getConfig(root,samp,'geomcell');
	logMsg(C,'geomcell started',1);
    %
    C = readMetadata(C);
    if (C.err==1)
        return
    end
    %
    C.geom  = fullfile(C.root,C.samp,'geom');
    if (exist(C.geom,'dir')~=7)
        mkdir(C.geom);
    end
    C.tifpath = fullfile(C.root,C.samp,'inform_data',...
        'Component_Tiffs');
    %
    if (opt==-1)
        return
    end
    %------------------------------------------
    % compute the boundary polygons for Tumor
    %------------------------------------------
    for n=1:numel(C.H.n)
        getCells(C,n,0);
    end
    %   
	msg = sprintf('geomcell processed %d fields',numel(C.H.n));
	logMsg(C,msg,1);
    %
end