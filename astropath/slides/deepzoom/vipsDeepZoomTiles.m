function vipsDeepZoomTiles(C, layer)
%%-----------------------------------------------------------
%% execute the vips command to create the deepzoom tiles
%% 2020-12-07   Alex Szalay
%%-----------------------------------------------------------
    %
    lyr = sprintf('%d',layer);
    logMsg(C,['Tiling for Layer(',lyr,') started']);
    %--------------------
    % build the command
    %--------------------
    wsi = [fullfile(C.zpath,'wsi\'),C.samp,'-Z9-L',lyr,'-wsi.png'];
    dst = [C.zdest,'\L',lyr];
    VIPS = '\\bki01\c$\apps\vips-dev-8.9\bin\';    
    cmd = [VIPS,'vips dzsave ',wsi,' ',dst,' --suffix .png --background 0',...
        ' --depth onetile --overlap 0 --tile-size 256'];
    %---------------------
    % execute the command
    %---------------------
    if (C.opt==0)
        status = system(cmd);
    else
        fprintf('%s\n',cmd);
    end    
    %
end