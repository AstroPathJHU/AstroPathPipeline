function runVIPSLoop(C)
%%-----------------------------------------------------------------------
%% run the VIPS command for the tiling, then delete the blank images.
%% Finally, renumber the zoom levels, fill in the missing small images
%% If there is an error, retry the VIPS loop
%%----------------------------------------------------------------------
    %
    if (C.opt>2)
        C.err = 1;
        return
    end
    logMsg(C,'Starting VIPS loop');
    %-----------------------------------------------------
    % Find out if there are any existing L*.dzi files
    % indicating that there was a crash during VIPS 
    %-----------------------------------------------------
    d=struct2table(dir(fullfile(C.zdest,'*.dzi')));    
    if (numel(d.bytes)>0)
        lastdzi = d.name{numel(d.bytes)};
        mx = replace(replace(lastdzi,'.dzi',''),'L','');
        if (mx<8)
            C.layers = mx+1:8;
        else
            return
        end
    end
    %
    for i= C.layers
        try
            vipsDeepZoomTiles(C,i);
        catch
            %
            % something went wrong, clean up temporary directories
            % and retry one more time from the last step
            %
            cleanupTempLayer(C,i);
            logMsg(C,sprintf('Restarting from layer %d',i),1);
            %
            % increase loop count
            %
            C.opt = C.opt+1;
            runVIPSLoop(C);
            return
            %
        end
        %
        try
            pruneDeepZoomTiles(C,i);
            patchDeepZoomTiles(C,i);
        catch
            logMsg(C,'ERROR: error in prune or patch',1);
            C.err=1;
            return
        end
        %
    end
end