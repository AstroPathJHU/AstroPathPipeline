function C = prepZoom(C,layer)
%function C = prepZoom(root1,samp,layer)
%%--------------------------------------------
%% prepare the images to be tiled and merged
%%   nx(n),ny(n) are the original tile positions
%%   qk(n) is the quadkey of the 16K tile
%%
%% Alex Szalay, 2019-04-05
%%--------------------------------------------
    %
    logMsg(C,'prepZoom');
    %
    %------------------------------
    % define base resolution level
    % also select marker layer
    %------------------------------
    C.zmax  = 9;
    C.layer = layer;
    C.bits  = 8;
    C.fmax = 50.0;
    %
    %----------------------------
    % read the metadata tables
    %----------------------------
    C = readMetadata(C);
    %
    C.xmax = floor(max(C.H.px))+C.fwidth;
    C.ymax = floor(max(C.H.py))+C.fheight;
    C.xmin = floor(min(C.H.px));
    C.ymin = floor(min(C.H.py));
    %-------------------------------
    % define the stiching tile size
    %-------------------------------
    C.tilex = 16384;
    C.tiley = 16384;
    C.buffer = 1536;
    %
    NX = floor(C.xmax/C.tilex)+1;
    NY = floor(C.ymax/C.tiley)+1;
    %--------------------
    % create directories
    %--------------------
    C.zoomroot = '\\bki02\f\zoom';
    %--------------------------------------------------
    % make zoom directory, if it does not exist
    %--------------------------------------------------
    zpath = fullfile(C.zoomroot,sprintf('Project%d',C.project));
    if (exist(zpath)==0)
        mkdir(zpath);
    end
    %
    C.zoompath = fullfile(zpath,C.samp);
    if (exist(C.zoompath)==0)
        mkdir(C.zoompath);
    end    
    %----------------------------
    % define path to TIFF images
    %----------------------------
    C.tiffext  = sprintf('_component_data.tif');
    C.tiffpath = [C.root '\' C.samp '\inform_data\Component_Tiffs\'];
    %{
    lumi = fullfile('\\bki02\f\lumi',sprintf('Project%d',C.project));
    if (exist(lumi)==0)
         mkdir(lumi);
    end
    %
    C.lumipath = fullfile(lumi,C.samp);
    if (exist(C.lumipath)==0)
         mkdir(C.lumipath);
    end
    %}
    %-----------------------------------------------------    
    % make buf,big, zoom directories if they do not exist
    %-----------------------------------------------------
    subpath = {'big','0','1','2','3','4','5','6','7','8','9'};
    for n=1:numel(subpath)
        zpathn = fullfile(C.zoompath, subpath{n});
        if (exist(zpathn)==0)
            mkdir(zpathn);
        end      
    end
    %
    C.scale = [1,1,1,1,1.1,1.2,1.5,2.25,3.375];
    %---------------------------
    % create tile array layout
    %---------------------------
    n=1;
    for i=1:NX
        for j=1:NY
            C.nx(n) = i;
            C.ny(n) = j;
            C.fname{n} = [sprintf('%s-Z%d-L%d-X%d-Y%d-big.png',...
                 C.samp,C.zmax,0,i-1,j-1)];
            n = n+1;
        end
    end   
    %
end


