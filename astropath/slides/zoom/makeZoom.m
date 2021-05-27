function C = makeZoom(C)
%%------------------------------------------------------------
%% create an array of DAPI thumbnail images at a given zoom level
%%
%% Alex Szalay, 2019-04-05
%%------------------------------------------------------------
    %
    t0 = clock();
    logMsg(C,'makeZoom');    
    %---------------------
    % do the stitching
    %---------------------
    for n=1:numel(C.nx)
        stitchZoom(C,n);
    end
    %
    s=sprintf('makeZoom finished in %f sec',etime(clock(), t0));
    logMsg(C,s);
    %
end

