function mergeZoom(C)
%%---------------------------------------------
%% at the lower zoom levels (3,2,1) merge the images
%% into bigger tiles, and adjust qkey accordingly
%%---------------------------------------------
    %
    t0 = clock();    
    logMsg(C.samp,'mergeZoom');
    %
    mm = sprintf('-Z3-L%d*.png',C.layer);
    f = [C.zoompath,'big\',C.samp,mm];
    d = dir(f);
    %
    % copy the Z3 files over to their directory
    %
    for n=1:numel(d)
        f1 = [C.zoompath,'big\',d(n).name];
        f2 = replace(f1,'-big.png','.png');
        f2 = replace(f2,'big','3');
        %fprintf('%s=>%s\n',f1,f2);
        copyfile(f1,f2);
    end
    %
    %return
    %
    mergeLevel(C,2);
    mergeLevel(C,1);
    mergeLevel(C,0);
    %
    s=sprintf('mergeZoom finished in %f sec',etime(clock(), t0));
    logMsg(C.samp,s);
    %
end
