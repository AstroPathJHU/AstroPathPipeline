function [T,P,a] = getTumorLayer(C,n)
%%----------------------------------------------------------------
%% Get the boundary information from a field from the InForm
%% tissue segmentation image. Tumor tissue has the value 0.
%% Use the adjacency matrix to determine the correct hierarchy 
%% inner and outer rings. Return the results in two forms,
%% P is the Paths object in clipper format, T is the table of
%% 
%%
%% Alex Szalay, Baltimore, 2019-03-03
%%----------------------------------------------------------------
    %
    T = [];
    P = [];
    %
    h = C.H(n,:);
    hf = replace(h.file{1},'.im3','_component_data_w_seg.tif');
    path = [C.root,'\',C.samp,'\inform_data\component_Tiffs\'];   
    f  = [path,hf];
    %
    try
        a = imread(f,9);
    catch
        fprintf('WARNING: missing file: %s\n',f);
        return
    end
    %
    [B,L,N,A] = bwboundaries(a==0);
    clear L
    P = boundary2path(B);
    %
    if (numel(P)==0)
        return
    end
    %
    h  = C.H(n,:);
    P = translatePaths(P,h.px,h.py);
    %
    [c,p] = find(A==1);
    %
    % p: parents   -- outer rings
    % c: children  -- inner rings inside parent
    % 
    r  = 1:N;
    ir = unique(c)';
    or = r(~ismember(r,c));
    %  
    for i=1:numel(or)
        %
        % for each k there is an outer ring
        %
        k  = or(i);
        out = ['POLYGON (',path2char(P(k))];
        %
        ck = c(p==k);
        if (numel(ck)>0)
            %
            %for each m there is an inner ring
            %
            for j=1:numel(ck)
                %
                m = ck(j);
                %
                P(m).x = flipud(P(m).x);
                P(m).y = flipud(P(m).y);
                %
                % inner rings must have >4 points
                %
                if (numel(P(m).x)>4)
                    out = [out,',',path2char(P(m))];
                end
                %
            end
            %
        end
        %
        E(i) = int32(n);
        K(i) = i;
        S{i} = [out,')'];
        %
    end
    %
    T = table(E',K',S');
    T.Properties.VariableNames = {'n','k','poly'};
    %
end





