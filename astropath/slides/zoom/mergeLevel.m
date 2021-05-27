function u = mergeLevel(C,zoom)
%%-----------------------------------------------------
%% take all the images at the zoom level n, and merge
%% them into their common parents and save
%%-----------------------------------------------------
    %
    zz = sprintf('%d',zoom);
    zpath = [C.zoompath,'big\'];
    mm = sprintf('-L%d',C.layer);
    ff = [zpath,C.samp,'-Z',zz,mm,'*.png'];
    d  = dir(ff);
    %
    % find the parent and child codes for each tile
    %
    tile = 256;
    nx   = 2^(3-zoom);
    siz  = tile/nx;
    %
    for n=1:numel(d)
        f{n} = d(n).name;
        %fprintf('%s\n',f{n});
        v = getParts(d(n).name);
        z(n) = v(1);
        m(n) = v(2);
        x(n) = v(3);
        y(n) = v(4);
        %
        px(n) = floor(x(n)*siz/256);
        py(n) = floor(y(n)*siz/256);
        pz(n) = 3*nx*py(n) + px(n)+1;
        iz(n) = mod(x(n),nx)+2*mod(y(n),nx);
        %
    end
    %
    % put things in a table for easy access
    %
    u = table(f',z',m',x',y',iz',px',py',pz');
    u.Properties.VariableNames = {'name','zoom','layer',...
        'x','y','iz','px','py','pz'};
    u = sortrows(u,{'pz','iz'});
    clear f x y px py pz iz
    %
    % now merge the children of the same parent
    %
    pts = unique(u.pz);
    for i=1:numel(pts)
        pt = pts(i);
        xx = u.px(find(u.pz==pt,1));
        yy = u.py(find(u.pz==pt,1));
        children = find(u.pz==pt);
        %
        % stack all the children
        %
        if (C.bits==16)
            T = uint16(zeros(tile,tile));
        elseif (C.bits==8)
            T = uint8(zeros(tile,tile));
        end
        for j=1:numel(children)
            ch = children(j);
            ff = [zpath,u.name{ch}];
            im = imread(ff);
            cx = mod(u.x(ch),nx)*siz;
            cy = mod(u.y(ch),nx)*siz;
            T(cy+1:cy+siz,cx+1:cx+siz)=im;
        end
        %
        mx = max(T(:));
        if (max(T(:))>3)
            %
            zz = sprintf('%d',zoom);
            f2 = sprintf('-L%d-X%d-Y%d',u.layer(1),xx,yy);
            zp = [C.zoompath,zz,'\'];
            if (exist(zp)==0)
                mkdir(zp);
            end
            f = [zp,C.samp,'-Z',zz,f2,'.png'];
            %---------------------------------------------
            %fprintf('(%f)=>%s\n',mx,f);
            imwrite(T,f);
        end
        %
    end
    %
end
