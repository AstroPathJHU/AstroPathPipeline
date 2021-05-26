function [Q,T] = getQPTiff(C)
%%-----------------------------------------------------------------
%% Get the relevant coordinate information from the QPTIFF file.
%% All distances are converted to microns.
%%
%% Alex Szalay, Baltimore, 2018-07-08
%%-----------------------------------------------------------------
    %
    logMsg(C,mfilename);
    %
    fpath = [C.root,'\',C.samp,'\Im3\',C.scan,'\',C.samp,...
        '_', C.scan '.qptiff'];
    %
    iq = imfinfo(fpath);
    %
    % find a reasonable magnification for the color composite
    %
    n = 7;
    nw = 100000;
    for i=1:numel(iq)
        if (iq(i).SamplesPerPixel==1 & iq(i).Width<nw)
            n = i;
            nw = iq(i).Width;
            if (nw<2000)
                break;
            end
        end
    end
    %

    Q.qplayer = n;
    Q.inf     = iq(n);
    Q.XResolution = iq(n).XResolution;
    Q.YResolution = iq(n).YResolution;    
    Q.ResolutionUnit = iq(1).ResolutionUnit;
    Q.XPosition = iq(1).XPosition;
    Q.YPosition = iq(1).YPosition;
    Q.qpscale = Q.XResolution; % in qpixels/micron
    Q.apscale = iq(1).XResolution;
    if (strcmp(Q.ResolutionUnit,'Centimeter'))
        Q.ResolutionUnit = 'Micron';
        Q.XPosition = Q.XPosition*1E4;
        Q.YPosition = Q.YPosition*1E4;
        Q.qpscale   = Q.XResolution/1E4;
        Q.apscale   = Q.apscale/1E4;
    end
    %
    Q.iq    = iq;
    Q.fname = fpath;
    %---------------------------
    % build a color image
    %---------------------------
    Q.img = mixQP(fpath,n,1);
    %
    % pack the database information into a table
    %
    jpgname = [C.samp '_qptiff.jpg'];
    imgbytes = '00001234';
    sampleid = 0;
    %
    T =  table(sampleid,string(C.samp),...
        string(Q.ResolutionUnit),Q.XPosition,Q.YPosition,...
        Q.XResolution,Q.YResolution,...
        Q.qpscale,Q.apscale,string(jpgname),string(imgbytes));
    T.Properties.VariableNames = {'SampleID','SlideID',...
        'ResolutionUnit','XPosition','YPosition',...
        'XResolution','YResolution','qpscale','apscale','fname','img'};
    %
end


function img = mixQP(f,n,opt)
%%---------------------------------------------
%% build a false color image from the 5 layers of the QPTIFF.
%% f is the filename, n is the first layer to start at,
%% color=0 specifies monochrome, other: RGB
%%
%% Alex Szalay, Baltimore, 2019-02-01
%%---------------------------------------------
    %
    img = single(imread(f,n));
    %
    if (opt==0)
        img = im2uint8(img);
        return
    end
    %
    % mixing matrix
    %{    
    DAPI:	(  0,  0, 255)
	FITS:	(  0,255,   0)
	CY3:	(255,255,   0)
	T-red: 	(255,128,   0)
	CY5:	(255,  0,   0)
    %}
    %
    % take the current HALO color model
    %
    mx = [  0.0, 0.0, 1.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.5, 0.0;
            1.0, 0.0, 0.0, 0.0, 0.0]/120;
    %
    % use single precision with lazy fetch for better memory usage
    % loop through the layers, mix and add
    %
    Q = repmat(0*img,[1,1,3]);
    for i=1:5
        if (i>1)
            img = single(imread(f,n+i-1));
        end
        for j=1:3
            Q(:,:,j) = Q(:,:,j) + mx(j,i)*img;
        end
    end
    %
    % convert to standard RGB
    %
    img = im2uint8(Q);
    %
end

