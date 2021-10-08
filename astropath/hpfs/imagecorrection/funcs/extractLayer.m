function d = extractLayer(path, sample, layer)
%%------------------------------------------------------
%% Take a directory of flatfielded and warped images in the raw
%% format, and extract a particular layer. Save it as a binary
%% file with an fwNN extension, where NN is the decimal number of layer.
%% Write the results back to the same directory. This would be .fw01
%% for the DAPI layer.
%%
%% Alex Szalay, Baltimore, 2018-07-29
%% Last Edit: Benjamin Green, Baltimore, 2021-04-06
%%
%% Usage: extractLayer('F:\dapi','M41_1',layer);
%%    or: extractLayer('F:\dapi','*',layer);
%%------------------------------------------------------
    %
    tic;  
    %
    % get the shape parameters
    %
    [ll, ww, hh] = get_shape([path,'\',sample], sample);
    %
    % get the image names
    %
    layer = str2double(layer);
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
        extractCore(d(i),layer, ll, ww, hh);
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


function extractCore(dd, layer, ll, width, height)
%%----------------------------------------------
%% inner core of the parallel loop
%% Alex Szalay, Baltimore, 2018-07-29
%% Last Edit: Benjamin Green, Baltimore, 2021-04-06
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
        r = permute(reshape(r,ll, width, height),[3,2,1]);   
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

function [ll, ww, hh] = get_shape(path, sample)
%%------------------------------------------------------
%% get the shape parameters from the parameters.xml file found at
%% [path/sample.Parameters.xml]
%% Last Edit: Benjamin Green, Baltimore, 2021-04-06
%%
%% Usage: get_shape('F:\new3\M27_1\im3\xml', 'M27_1');
%%------------------------------------------------------
    p1 = fullfile(path, [sample,'.Parameters.xml']);
    mlStruct = parseXML(p1);
    params = strsplit(mlStruct(5).Children(2).Children.Data);
    ww = str2double(params{1});
    hh = str2double(params{2});
    ll = str2double(params{3});
end