function O = mergeBigTiles(C,varargin)
%%-----------------------------------------------
%% merge the 16Kx16K tiles into a full image for a given sample. 
%% The big tiles are in the root\sample\big directory. The merged
%% images are created in the same root but in the wsi subdirectory.
%% If the optional argument is different from 0, the command is printed,
%% otherwise it is executed.
%% Example:
%%  mergeBigTiles(C,1);
%%
%% 2020-05-14   Alex Szalay
%%----------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    dbg=0;
    O =[];
    %
    logMsg(C,'mergeBigTiles');
    C.zoom = 9;
    layer  = 1;
    %
    VIPS = '\\bki01\c$\apps\vips-dev-8.9\bin\';
    %
    %check if merge exists, if not create it
    %
    merge   = fullfile(C.zoompath,'wsi');
    if (exist(merge)==0)
        mkdir (merge);
    end
    blank = fullfile(C.zoomroot,'blank',sprintf('blank-Z%d-B8.png',C.zoom));
    %
    % get a list of files of the right zoom and layer
    %
    p = fullfile(C.zoompath,'big',...
        sprintf('%s-Z%d-L%d',C.samp,C.zoom,layer));
    %
    d = dir([p,'*.png']);
    if (numel(d)==0)
        msg = sprintf('ERROR: No tiles found in %s',...
            [C.root,'\',C.samp,'\big']);
        logMsg(C,msg);
        return
    end
    %
    for i=1:numel(d)
        f{i} = [d(i).folder,'\',d(i).name];
        v(i,:) = getParts(d(i).name);
    end
    %    
    mx = max(v(:,3));
    my = max(v(:,4));
    %
    list ='';
    for i=0:my
        for j=0:mx
            ix = find(v(:,3)==j & v(:,4)==i);
            if (isempty(ix))
                fname = blank;
            else
                fname = f{ix};
            end
            list = [list,fname,' '];
            if (dbg==1)
                fprintf('%s\n', fname);
            end
        end
    end
    %    
    cmdtmp = [VIPS,sprintf('vips arrayjoin "%s" ',list), merge,'\', ...
        sprintf('%s-Z%d-L%d-wsi.png --across %d',...
        C.samp,C.zoom, layer, mx+1)];
    lyr = sprintf('-L%d-',layer);
    for i=1:numel(C.layer)
        s = sprintf('-L%d-',C.layer(i));
        cmd = replace(cmdtmp,lyr,s);
        if (opt==0)
            status = system(cmd);
        else
            fprintf('%s\n',cmd);
        end
    end
    %
    msg = sprintf('mergeBigTiles generated %d layers',numel(C.layer));
    logMsg(C,msg);
    % 
end


function v = getParts(fname)
%%----------------------------------------
%% split the filename into parameters
%% 2020-05-14   Alex Szalay
%%----------------------------------------
    %
    f1 = replace(fname,'.','-');
    a = strsplit(f1,'-');
    a = a(2:5);
    %
    for i=1:numel(a)
        v(i) = str2num(a{i}(2:end));
    end
end

