function C =getImages(C,varargin)
%%---------------------------------------------
%% Read the first layer from both the QPTiff
%% and from the whole slide Astropath image.
%% scale both to be close to each other
%%
%% 2020-07-05  Alex Szalay
%%--------------------------------------------
    %    
    logMsg(C,'getImages started');
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    %-------------------------------------------------------------
    % C.ipscale = the ratio of the scales between the highest 
    % resolution WSI and QPTiff images. We scale down the WSI by 
    % C.ppscale = the closest integer to the QPTIFF. 
    % C.iqscale is how to 
    %-------------------------------------------------------------
    C.ipscale = C.pscale/C.apscale;
    C.ppscale = round(C.ipscale);
    C.iqscale = C.ipscale/C.ppscale;
    if (opt==-1)
        return
    end
    %----------------------------------------------
    % get the QPTIFF image, and rescale it to ipix
    %----------------------------------------------
    if (isfield(C,'Q')==0)
        C.Q = getQPTiff(C);
    end
    %
    qpname  = fullfile(C.root,C.samp,'Im3',C.scan,...
        [C.samp,'_',C.scan,'.qptiff']);
    try
        C.qimg  = imread(qpname,1);
    catch
        msg =sprintf('ERROR: cannot open %s', qpname,1);
        C.err=1;
        return
    end
    %
    %----------------------------------------
    % get WSI image on the real pixel scale
    %----------------------------------------
    %zoomroot = '\\bki02\f\zoom';
    zoomroot = '\\bki01\c$\data\data11\zoom';
    proj    = sprintf('Project%02d',C.project);
    zoomdir = fullfile(zoomroot,proj,C.samp,'wsi');
    wsiname = fullfile(zoomdir,[C.samp,'-Z9-L1-wsi.png']);
    try
        C.aimg  = imread(wsiname);
    catch
        msg =sprintf('ERROR: cannot open %s', wsiname,1);
        C.err=1;
        return
    end

    % Finally, we rescale the QPTiff by a fractional scale close to 1
    % C.iqscale = C.ipscale/C.ppscale. 
    %-------------------------------------------------------------
    C.aimg = imresize(C.aimg,1/C.ppscale);
    C.qimg = imresize(C.qimg,C.iqscale);
    %-----------------------------------
    % clip C.aimg to the size of C.qimg
    %-----------------------------------
    [na,ma] = size(C.aimg);
    [nq,mq] = size(C.qimg);
    if (nq<=na & mq<=ma)
        C.aimg = C.aimg(1:nq,1:mq);
    elseif (nq>=na & mq>=ma)
        C.qimg = C.qimg(1:na,1:ma);
    end
    %---------------------
    % show the two images
    %---------------------
    if (opt>0)
        close all
        figure(1);imshow(C.aimg);
        set (gca,'Ydir','normal');        
        figure(2);imshow(C.qimg);
        set (gca,'Ydir','normal');
        shg
    end
    %
end
    
