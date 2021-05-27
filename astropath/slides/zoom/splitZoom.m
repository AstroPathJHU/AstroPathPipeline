function splitZoom(C)
%%----------------------------------------------
%% split each image in the Z=N directory into
%% its four children
    %
    t0 = clock();
    logMsg(C.samp,'splitZoom');    
    %
    for level = 4:9
        %
        mm = sprintf('-Z%d',level);
        d = dir([C.zoompath,'big\',C.samp,mm,'*.png']);
        %
        for n=1:numel(d)
           splitBigTile(C,d(n),level);
        end
        %
    end
    %
    s=sprintf('splitZoom finished in %f sec',etime(clock(), t0));
    logMsg(C.samp,s);
    %
end


function splitBigTile(C,dn,level)
%%---------------------------------------
%% execute the core splitting function
%%---------------------------------------
    %
    ff = [dn.folder,'\', dn.name];
    try
        a  = imread(ff);
        %fprintf('\nInput:%s\n',ff);
    catch
        s = sprintf('cannot read %s\n',ff);
        return
    end 
    %
    nx = size(a,2)/256;
    ny = size(a,1)/256;
    nn = log(nx)/log(2);
    %
    v = getParts(dn.name);
    z  = v(1);
    m  = v(2);
    xx = nx*v(3);
    yy = ny*v(4);
    %
    % do all the subtiles
    %
    for iy = 1:ny
        y = 256*(iy-1);
        for ix=1:nx
            x = 256*(ix-1);
            f = [C.zoompath,sprintf('%d',level),'\',C.samp,...
              sprintf('-Z%d-L%d-X%d-Y%d.png',z,m,xx+(ix-1),yy+(iy-1))];
            b = a(y+1:y+256,x+1:x+256);
            %
            % only write the img if it is not empty
            %
            if (max(b(:))>3)
                %fprintf('   Out[%d,%d]:%s\n',ix-1,iy-1,f);
                imwrite(b,f);
            end
        end    
    end
    %
end
%