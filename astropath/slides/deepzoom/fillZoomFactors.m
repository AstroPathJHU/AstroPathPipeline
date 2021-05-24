function dz = fillZoomFactors(C,layer)
%%------------------------------------------
%% Create the missing low Zoom numbers
%% and fill them with the scaled images.
%% Runs on a single layer.
%%
%% 2020-12-07   Alex Szalay
%%------------------------------------------
    %
    lyr = sprintf('%d',layer);
    dz = struct2table(dir([C.zdest,'\L',lyr,'_files\Z*']));
    dz.zoom = cellfun(@str2num,replace(dz.name,'Z',''));
    %
    root = dz.folder{1};
    %
    mz = min(dz.zoom);
    %
    if (mz==0)
        return
    end
    %
    mpath = dz.name(dz.zoom==mz);
    mname = [root,'\',mpath{1},'\0_0.png'];
    img   = imread(mname);
    blank = zeros(256,256,class(img));
    %---------------------------------------------
    % check if this is 256x256
    % if not, copy it into the right image size
    %---------------------------------------------
    [n,m] = size(img);
    if (n~= 256 | m~= 256)
        a = blank;
        a(1:n,1:m)=img;
        imwrite(a, mname);
        img = a;
    end
    %--------------------------------------
    % create the additonal low zoom levels
	% all the way to Z0, and pack it into
	% a 256x256 image
    %--------------------------------------
    newzoom = (0:mz-1);
    %
    for zoom=newzoom
        newdir  = fullfile(root, sprintf('Z%d',zoom));
        newname = [newdir,'\0_0.png'];
        if (~exist(newdir))
            mkdir(newdir);
        end        
        %----------------------------------------
        % zoom the image by the relative factor
        % and write it to disk
        %----------------------------------------
        scale = 1/2^(mz-zoom);
		amp = 1.25^(mz-zoom);
        a = imresize(img,scale);
        [n,m] = size(a);
        b = blank;
        b(1:n,1:m)= uint8(amp*a);
        imwrite(b,newname);
        %
    end
    %
end

