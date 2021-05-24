function showOutlines(C,varargin)
%%----------------------------------------------------------
%% show the field layout with color-coded exposure times
%%----------------------------------------------------------
    %
    % field offsets and sizes all in microns
    %
    hold on
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %    
    axis equal
    box on
    pw = C.fwidth/C.pscale;  % in microns
    ph = C.fheight/C.pscale;  % in microns
    dx = 0.35*pw;
    dy = 0.35*ph;
    %
    tx = C.qpscale*(C.R.x-C.xposition+dx);
    ty = C.qpscale*(C.R.y-C.yposition+dy);
    %
    tw = C.qpscale*(pw-dx);
    th = C.qpscale*(ph-dy);
    %
    % show the field outlines
    %
    cb = [0.5,0.5,0.5];
    cf = [0.5,0.5,0.5];
    %
    cm = colormap(gray);
    em = table2array(C.E(:,2+opt));
    e1 = min(em);
    e2 = max(em);
    ix = 1+floor(200*(em-e1)/(e2-e1));
    N = numel(tx);
    for i=1:N        
        rect2poly(tx(i),ty(i),tx(i)+tw,ty(i)+th,cb,cm(ix(i)),1);
    end
    %
    x1 = min(tx)-dx;
    y1 = min(ty)-dy;
    x2 = max(tx)+tw+dx;
    y2 = max(ty)+th+dy;
    axis([x1 x2 y1 y2]);
    colorbar
    str = sprintf('[%s] layer %d',replace(C.samp,'_','-'),opt);
    title(str)        
    
    %
end



function p = rect2poly(x1,y1,x2,y2,cb,cf,flag)
%%----------------------------------------------------
%% draw a rectangle with the diagonal (x1,y1)-(x2,y2)
%%----------------------------------------------------
    %
    x = [x1,x2,x2,x1,x1];
    y = [y1,y1,y2,y2,y1];
    p = fill(x,y,cf);
    p.EdgeColor=cb;
    if (flag==0)
        p.FaceColor='none';
    end
    %
end
