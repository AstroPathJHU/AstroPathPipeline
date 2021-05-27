function C = scaleZoom(C)
%%--------------------------------------------
%% read in the scale 1 images and zoom them
%% as well as creating the quadtree address
%%
%%--------------------------------------------
    %
    t0 = clock();
    logMsg(C.samp,'scaleZoom');    
    %
    parfor n=1:numel(C.nx)
        try
            f = [C.zoompath,'buf\',C.fname{n}];
            %fprintf('.%s\n',f);
            a   = imread(f);
            buf = C.buffer;
            siz = C.tilex;
            sc  = 1.0;
            %
            % clip off buf, save original size
            %
            bb = a(buf+1:buf+siz, buf+1:buf+siz);
            ff = replace(f,'buf','big');
            imwrite(bb,ff);
            zz = sprintf('Z%d',C.zmax);
            %
            
            for i=1:6
                %
                % rescale by factors of 2, clip and save
                %
                buf = buf/2;
                siz = siz/2;
                sc  = sc/2;
                %
                bb = imresize(a,sc,'Method','bicubic',...
                    'Antialiasing', true);                
                bb = bb(buf+1:buf+siz, buf+1:buf+siz);
                %
                ff = replace(f,zz,sprintf('Z%d',C.zmax-i));
                ff = replace(ff,'buf','big');
                imwrite(bb,ff);
                %
            end
            %
        catch
            % quadkey exists, but no image written, as it was empty
        end
    end
    %
    s=sprintf('scaleZoom finished in %f sec',etime(clock(), t0));
    logMsg(C.samp,s);
    %
end

