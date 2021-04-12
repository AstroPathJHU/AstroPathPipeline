function d = extractLayer(path,sample,layer)
%%------------------------------------------------------
%% Take a directory of flatfielded and warped images in the raw
%% format, and extract a particular layer. Save it as a binary
%% file with an fwNN extension, where NN is the decimal number of layer.
%% Write the results back to the same directory. This would be .fw01
%% for the DAPI layer.
%%
%% Alex Szalay, Baltimore, 2018-07-29
%%
%% Usage: extractLayer('F:\dapi','M41_1',layer);
%%    or: extractLayer('F:\dapi','*',layer);
%%------------------------------------------------------
    %
    tic;    
    %
    % get the image names
    %
    ww = 1344;
    hh = 1004;
    %
    layer = str2num(layer);
    %
    if (strcmp(sample,'*'))
        p1 = path;
    else
        p1 = [path '\' sample];
    end
    %
    p = [p1,'\**\*.fw'];
    d = dir(p);
    %    
    if (numel(d)==0)
        disp(sprintf('No fw files found in path %s',p1));
        return
    end
    %
    % run the parallel loop
    %
    if isempty(gcp('nocreate'))
        numcores = feature('numcores');
        usecores = 4;
        evalc('parpool(usecores)');
    end
    %
    parfor i=1:numel(d)
        extractCore(d(i),layer, ww, hh);
    end
    %
    dt = toc;
    disp(sprintf('    %d files in %f sec',numel(d), dt));
    %
    % Ben edit: I suppressed these because I could not turn off the message
    % to the console, so the pool closing messages displayed in the lof
    % file. The pool should still shut down when Matlab finishes the .exe
    % shut down parallel pool
    %
    %poolobj = gcp('nocreate');
    %delete(poolobj);
    %
end


function extractCore(dd, layer, width, height)
%%----------------------------------------------
%% inner core of the parallel loop
%% Alex Szalay, Baltimore, 2018-07-29
%%----------------------------------------------
    ext = sprintf('.fw%02d',uint16(layer));
    f1 = [dd.folder '\' dd.name];
    f2 = replace(f1,'.fw',ext);
    %
    fd = fopen(f1,'r');
    r = uint16(fread(fd,'uint16'));    
    fclose(fd);
    %
    img = [];
    try
        r = permute(reshape(r,35,width,height),[3,2,1]);   
        img = r(:,:,layer);
    catch
        f1
        layer
    end
    %        
    fd = fopen(f2,'w');
    fwrite(fd,img(:),'uint16');
    fclose(fd);
    %
end


