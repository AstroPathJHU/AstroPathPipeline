function stitchZoom(C,n)
%%------------------------------------------------------------
%% take the struct containing the hpfid's 
%% and write unit8 image layers for the nth tile
%% 2018-08-20   Alex Szalay
%% 2020-06-26   Alex Szalay: modified code to do all 8 layers
%%                  in the same pass for speed
%%------------------------------------------------------------
    dbg = 1;
    if (dbg==1)
        fprintf('Tile(%d):(%d,%d)\n',n, C.nx(n)-1,C.ny(n)-1);
    end
    %----------------------
    % resolve tile numbers
    %----------------------
    nx = C.nx(n);
    ny = C.ny(n);
    %-----------------------------------------------------
    % get the offsets and limits from the min and max, 
    % the nearest thousand measured in pixel coordinates
    %-----------------------------------------------------
    sizey = C.tiley+2*C.buffer;
    sizex = C.tilex+2*C.buffer;
    %
    %-------------------
    % set the tile origin
    %-------------------
    tx = (nx-1)*C.tilex;
    ty = (ny-1)*C.tiley;
    %{
    fprintf('...(%d,%d)-(%d,%d) == (%d,%d)-(%d,%d)\n',...
        tx,ty,tx+sizex,ty+sizey, C.xmin,C.ymin, C.xmax,C.ymax);
    %}
    if (tx+sizex<C.xmin | ty+sizey<C.ymin | tx>C.xmax | ty > C.ymax)
        return
    end
    %--------------------------
    % create the tile buffer
    %--------------------------
    if (C.bits==16)
        T = zeros(sizey,sizex,numel(C.layer),'uint16');
    elseif (C.bits==8)
        T = zeros(sizey,sizex,numel(C.layer),'uint8');
    end    
    %----------------------------------
    % get the fractional pixel shifts
    %----------------------------------
    dx = C.H.px-floor(C.H.px);
    dy = C.H.py-floor(C.H.py);
    %------------------------
    % get the source origin
    %------------------------
    ax = floor(C.H.mx1-C.H.px);
    ay = floor(C.H.my1-C.H.py);
    %-----------------------------
    % get the destination origin
    %-----------------------------
    bx = floor(C.H.mx1+1)-tx+C.buffer;
    by = floor(C.H.my1+1)-ty+C.buffer;
    %----------------
    % get the sizes
    %----------------
    sx = floor(C.H.mx2-C.H.mx1+1);
    sy = floor(C.H.my2-C.H.my1+1);
    %
    if (dbg==1)
        fprintf('numel(C.layer)=%d\n',numel(C.layer));
    end
    for i=1:numel(C.H.n)
        if (dbg==1)
            fprintf('<%d,%d>\n',n,i);
        end
        %----------------------------------------------------
        % test whether it intersects with the tile rectangle
        %----------------------------------------------------
        if ((C.H.px(i)< tx-C.fwidth) ...
                | (C.H.px(i)>tx+C.tilex)...
                | (C.H.py(i)<ty-C.fheight) ...
                | (C.H.py(i)>ty+C.tiley) )
            continue
        end
        %
        img = getField(C,i);
        if (isempty(img))
            continue
        end
        %-----------------
        % translate field
        %-----------------
        img = imtranslate(img,[dx(i),dy(i)],'cubic'); 
        %
        fprintf('[d] <%d,%d> %d, %d, %d, %d\n',...
            n,i,by(i)+1,by(i)+sy(i),bx(i)+1,bx(i)+sx(i));
        fprintf('[s] <%d,%d> %d, %d, %d, %d\n',...
            n,i,ay(i)+1,ay(i)+sy(i),ax(i)+1,ax(i)+sx(i));
        %
        T(by(i)+1:by(i)+sy(i),bx(i)+1:bx(i)+sx(i),:) =...
            img(ay(i)+1:ay(i)+sy(i),ax(i)+1:ax(i)+sx(i),C.layer);
        %
    end
    %--------------------
    % clip off the edges
    %--------------------
    buf = C.buffer;
    siz = C.tilex;
    T   = T(buf+1:buf+siz, buf+1:buf+siz,:);
    %---------------------------------
    % write the full buffer, but only 
    % if the clipped one is not empty
    %---------------------------------
    if (max(T(:))>0)
        f1 = fullfile(C.zoompath ,'big', C.fname{n});
%        f2 = replace(f1,'buf','big');       
        for i=1:numel(C.layer)
            g1 = replace(f1,'-L0-',sprintf('-L%d-',C.layer(i)));            
            savepng1(T(:,:,i),g1,3);
        end
        %{
        %
        % save the clipped image at the original level
        %
        ff = [C.zoompath ,'big\', replace(C.fname{n},'buf','big')];
        ff = replace(ff,'.png','.b8');
        fprintf('...%s\n',ff)
        fp = fopen(ff,'w');
        if (C.bits==16)
            fwrite(fp,t,'uint8');
        else
            fwrite(fp,t,'uint8');
        end
        fclose(fp);
        %
        % rescale, clip and save at every zoom level
        %
        for i=1:9
            buf = buf/2;
            siz = siz/2;
            sc  = sc/2;
            %
            t = imresize(T,sc,'Method','bicubic','Antialiasing', true);                
            t = t(buf+1:buf+siz,buf+1:buf+siz);
            f = replace(ff,'Z9',sprintf('Z%d',9-i));
            fprintf('....%s\n',f)
            %
            fp =fopen(f);
            if (C.bits==16)
                fwrite(fp,uint16(C.scale(i)*t),'uint8');
            else
                fwrite(fp,uint8(C.scale(i)*t),'uint8');
            end
            fclose(fp);
            %
        end
        %}
    end
    %
end

