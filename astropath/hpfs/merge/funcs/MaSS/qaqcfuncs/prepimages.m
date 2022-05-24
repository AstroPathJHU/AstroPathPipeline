%% prepimages
%% Created by: Benjamin Green
%% ---------------------------------------------
%% Description
% prepare the images by multiplying by the corresponding color vectors and
% adding the segmentation.
%% Input
% im = image column vector where the first column is dapi
% c_map = color matrix for the corresponding image
% im_size = [h w] of the returned image
% scol = the shade of red for the segmentation map
% seg = the segmentation column vector
%% Ouput
% im_dapi = color image in matrix format with segmentation; with dapi
% im_nodapi = color image in matrix format with segmentation; no dapi
%% ---------------------------------------------
%%
function [im_dapi, im_nodapi] = prepimages(im, c_map, im_size, scol, seg)
%
% create dapi images first
%
im_dapi = 180 * sinh(1.5 * im) * c_map;
im_dapi(seg,:) = repmat([scol 0 0], length(seg),1);
im_dapi = uint8(im_dapi);
im_dapi = reshape(im_dapi,[im_size, 3]);
%
% create no dapi images next
%
im = im(:,2:end);
c_map = c_map(2:end,:);
im_nodapi = 180 * sinh(1.5 * im) * c_map;
if ~isempty(seg)
    im_nodapi(seg,:) = repmat([scol 0 0], length(seg),1);
end
im_nodapi = uint8(im_nodapi);
im_nodapi = reshape(im_nodapi,[im_size, 3]);
%
end
