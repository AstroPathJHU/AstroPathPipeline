%%  mkindvphenim
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% This fucntion creates image mosiacs for lineage markers and dual
%%% expressions with expression markers
%% --------------------------------------------------------------
%%
function mkindvphenim(d, mycol, imageid, im,...
    ims, Markers, im_full_color, im_full_color_seg)
%
[Image,expr] = getset(Markers, imageid);
%
% get locations of segmentation
%
seg = reshape(ims,[],1);
seg = find(seg > 0);
scol = uint8(255 * .65);
%
for M = 1:length(Image.all_lineages)
    %
    % lineage image with dapi
    %
    current_marker = Image.all_lineages{M};
    %
    % for additional markers, aka dual lineage expression, we need to set
    % up special image and color vectors. Otherwise just use the specified
    % layer
    %
    if ismember(current_marker, Markers.add)
        %
        % get the position vectors
        %
        [SW, EW] = resolveMultiLin(current_marker, Markers);
        %
        % get the image columns and color
        %
        im_lineage_dapi = [im(:,1), im(:,EW), im(:,SW)];
        cc = [mycol.all(1,:); mycol.all(EW,:);  mycol.all(SW,:)];
        compartment = 'NULL';
        %
    else
        %
        % get the image columns and color
        %
        im_lineage_dapi =[im(:,1), im(:,Image.layer(M))];
        cc = [mycol.all(1,:); 1 1 1];
        compartment = Image.Compartment(M);
    end
    %
    % get the lineage image with and without dapi in color
    %
    [im_lineage_dapi_color, im_lineage_nodapi_color] = ...
        prepimages(im_lineage_dapi, cc, imageid.size, scol, seg);
    [im_lineage_dapi_color_noseg, im_lineage_nodapi_color_noseg] = ...
        prepimages(im_lineage_dapi, cc, imageid.size, scol, []);
    %
    % get the locations of the positive cells
    %
    ii = strcmp(d.fig.Phenotype,Image.all_lineages{M});
    data.ii = ii;
    data.pos = d.fig(ii,:);
    x = data.pos.CellXPos;
    y = data.pos.CellYPos;
    xy = [x y];
    data.xy = xy;
    %
    if height(data.pos) > 1
        %
        % create single color image for phenotyped image with dapi
        %
        create_color_images(im_lineage_dapi_color, imageid.outABlin{M},...
            Image,im_full_color, im_full_color_seg, data, d.fig, ...
            im_lineage_nodapi_color, im_lineage_dapi_color_noseg, ...
            im_lineage_nodapi_color_noseg, compartment)
        %
        % make images for expression marker & lineage coexpression
        %
        for t = 1:length(expr.namtypes)
            %
            % get the locations of the positive cells
            %
            ii = strcmp(d.fig.Phenotype,Image.all_lineages{M}) &...
                d.fig.(lower(expr.namtypes{t}));
            data.ii = ii;
            data.pos = d.fig(ii,:);
            xy = [data.pos.CellXPos data.pos.CellYPos];
            data.xy = xy;
            %
            if height(data.pos) > 1
                %
                % put image together for expr lin, dapi, segmentation
                %
                ly = expr.layers(t);
                ime = im(:,ly);
                imela = [im_lineage_dapi,ime];
                cc1 = [cc; mycol.all(ly,:)];
                %
                [imel, imelnd] = ...
                    prepimages(imela, cc1, imageid.size, scol, seg);
                [imel_noseg, imelnd_noseg] = ...
                    prepimages(imela, cc1, imageid.size, scol, []);
                %
                create_color_images(imel, [imageid.outABcoex{M},...
                    expr.namtypes{t}], Image, im_full_color,...
                    im_full_color_seg, data, d.fig,...
                    imelnd, imel_noseg, imelnd_noseg, compartment);
            end
        end
    end
end
%
%
end
