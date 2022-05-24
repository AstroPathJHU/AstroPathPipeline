%%  mkexprim
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% This fucntion creates image mosiacs for expression markers
%% --------------------------------------------------------------
%%
function mkexprim(mycol, imageid, im,...
    Markers, im_full_color, im_full_color_seg)
%
[Image,~] = getset(Markers,imageid);
%
% get the segmentations for each expression variable, including dual 
% segmentations 
%
[ims, expr, d2] = getSegMaps(imageid, Markers);
%
for t = 1:length(expr.namtypes)
    %
    if (width(d2{t}) ~= 7)
        continue
    end
    %
    compartment = expr.compartment(t);
    %
    % put image together fused for expr, dapi, segmentation
    %
    ly = expr.layer(t);
    %
    imea = [im(:,1),im(:,ly)];
    cc = [mycol.all(1,:); 1 1 1];
    %
    seg = ims(:,t);
    seg = find(seg > 0);
    scol = uint8(255 * .65);
    % ---------------------------------------------------------------------
    % need to recreate segmentation vectors based on Markers.nsegs %%%%%%%
    % ---------------------------------------------------------------------

    [ime, imend] = ...
        prepimages(imea, cc, imageid.size, scol, seg);
    [ime_noseg, imend_noseg] = ...
        prepimages(imea, cc, imageid.size, scol, []);
    %
    % get the positive cells
    %
    ii = d2{t}.ExprPhenotype;
    %
    % set up struct
    %
    data.ii = ii;
    data.pos = d2{t}(ii,:);
    x = data.pos.CellXPos;
    y = data.pos.CellYPos;
    xy = [x y];
    data.xy = xy;
    %
    if height(data.pos) > 1
        %
        create_color_images(ime, imageid.outABexpr{t},...
            Image,im_full_color, im_full_color_seg, data, d2{t}, imend, ...
            ime_noseg, imend_noseg, compartment);
        %
    end
end
end
