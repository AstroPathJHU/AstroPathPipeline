function T = cellBoundary(C,n,ctype)
%%----------------------------------------------------------------
%% Get the membrane information for a field from the InForm
%% cell segmentation images. flag ==1 : write the data to disk.
%% celltype:
%%  0 : membrane tumor
%%  1 : membrane immune
%%  2 : nucleus tumor
%%  3 : nucleus immune
%%
%% Alex Szalay, Baltimore, 2019-03-03
%%----------------------------------------------------------------
    %
    o =[];
    T =[];
    %
    h = C.H(n,:);
    f  = [C.tifpath,'\',replace(h.file{1},...
        '.im3','_component_data_w_seg.tif')];
    %
    if (ctype==0)
        layer = 13;
    elseif (ctype==1)
        layer = 12;
    elseif (ctype==2)
        layer = 11;
    elseif (ctype==3)
        layer = 10;
    else
        logMsg(C,'ERROR: Illegal cell type');
        C.err=1;
        return
    end
    %
    try
        a = imread(f,layer);
    catch        
        msg = sprintf('WARNING: missing file: %s',f);
        logMsg(C,msg);
        return
    end
    %----------------------------
    % get unique set of labels
    %----------------------------
    u = unique(a(:));
    u(1) = [];
    %
    bn = [];
    r = regionprops(a,{'BoundingBox','Image'});
    k = 1;
    for i=1:numel(r)
        if( ~isempty(r(i).Image))
            box = round(r(i).BoundingBox);
            box(1:2) = box(1:2)+floor([h.px,h.py]);
            box = int32(box);
            pb  = bwboundaries(r(i).Image);
            pb  = int32(pb{1});
            pb  = pb(:,[2,1]);
            %
            z = bsxfun(@plus,int32(pb),box(1:2));
            if (size(z,1)<=4)
               continue
            end
            %
            try
                s  = char(join(join(string(z))',','));
                s  = ['POLYGON ((',s ,'))'];
            catch
                s ='';
            end
            bn(k)   = i;
            bb(k,:) = box;
            bs{k}   = s; 
            %
            k = k+1;
        end
    end
    %
    if (numel(bn)>0)
        type  = int32(0*bn + ctype);
        field = int32(0*bn+n);        
        T = table(field',type',bn',bb(:,1),bb(:,2),bb(:,3),bb(:,4),bs');
        T.Properties.VariableNames = {'field','ctype','n',...
            'x','y','w','h','poly'};
    end
    %
end





