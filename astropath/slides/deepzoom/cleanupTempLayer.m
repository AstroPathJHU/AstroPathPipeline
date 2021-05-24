function cleanupTempLayer(C,layer)
%%-------------------------------------------
%% will try to clean up the temp files from
%% an aborted VIPS run
%%-------------------------------------------
    %
    dname = fullfile(C.zdest,sprintf('L%d-*',layer));
    d = dir(dname);
    if (d.isdir==1)
        logMsg(C,sprintf('Removing dir %s',dname));
        rmdir(dname,'s');
    end
    %
end