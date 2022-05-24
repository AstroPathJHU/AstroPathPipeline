%%  getSegMaps
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
% This fucntion gets the segmentation maps for each expr variable
% including multiple segmentations. Returns a column vector with binary
% segmentation, a data struct with two fields (nametypes and layer), and
% modifies the outABexpr class of imageid. Also modifies the expression
% marker columns of the d.fig table to separate multiple segmentations.
%% Output
% ims = a column vector where each column is the binary map for the
% segmentation
% expr.namtypes = cell array of expression markers including additional
% segmentations for multiple segmentations on the same AB
% expr.layer = numeric array for the layer in the compenet tiff of each
% segmentation
% imageid.outABexpr = cell array of destination paths for the images
% d.fig = columns for multiple segmentations divided into parts
%% --------------------------------------------------------------
%%
function [ims, expr, d2] = getSegMaps(imageid, Markers)
%
% get antibody folder names
%
AB_fdnames = cellfun(@(x)extractBetween(x,'\Phenotype\',['\',imageid.id]),...
    imageid.outABexpr,'Uni',0);
AB_fdnames = [AB_fdnames{:}];
expr.namtypes = AB_fdnames;
expr.layer = imageid.exprlayer;
[~, loc] = ismember(expr.layer, Markers.Opals);
expr.compartment = Markers.Compartment(loc);
%
% get the cell x and y positions from each segmentation map
%
seg_types = [Markers.seg,Markers.altseg];
layers = length(Markers.Opals) + 2;
filnm = [imageid.id,'cell_seg_data'];
%
xy_seg = cellfun(@(x) get_pheno_xy(filnm,x,imageid.wd,layers),seg_types,'Uni',0);
xy_expr = cellfun(@(x) get_pheno_xy(filnm,x,imageid.wd,layers),AB_fdnames,'Uni',0);
d2 = xy_expr;
%
% convert to subscripts so we can perform matrix comparisons
%
loc_seg = cellfun(@(x) sub2ind(imageid.size,...
    x.CellYPosition,x.CellXPosition), xy_seg,'Uni',0);
num_seg = cellfun('length',loc_seg);
%
loc_expr = cellfun(@(x) sub2ind(imageid.size,...
    x.CellYPosition,x.CellXPosition), xy_expr,'Uni',0);
num_expr = cellfun('length',loc_expr);
%
% for each expression marker see which segmentation map it fits in
%
ims = zeros(imageid.size(1) * imageid.size(2),length(xy_expr));
%
for i1 = 1:length(xy_expr)
    %
    % find segmentations that have the same number of cells
    %
    idx = find(num_expr(i1) == num_seg); 
    %
    % if more than one segmentation type has the same number of cells
    % compare positions to determine current segmenation map
    %
    if length(idx) > 1
        for i2 = 1:length(idx)
            val = loc_expr{i1} == loc_seg{idx(i2)};
            if sum(val) == length(loc_expr{i1})
                c_seg = seg_types{idx(i2)};
                break
            end
        end
    elseif length(idx) == 0
        continue
    else
        c_seg = seg_types{idx};
    end
    %
    % read in that segmentation map and convert it to a column vector
    %
    folds = [imageid.wd,'\Phenotyped\',c_seg];
    im_name = [imageid.id,'binary_seg_maps.tif'];
    im_full = fullfile(folds,im_name);
    %
    seg_im = imread(im_full,4);
    ims(:,i1) = reshape(seg_im,[],1);
    %
    % make binary columns for d2
    %
    ii = expr.layer(i1) == Markers.Opals;
    AB = Markers.all(ii);
    expr.layer(i1) = find(ii) + 1;
    expr.compartment(i1) = Markers.Compartment(ii);
    d2{i1}.ExprPhenotype = strcmp(d2{i1}.Phenotype, AB);
    d2{i1}.CellXPos = d2{i1}.CellXPosition;
    d2{i1}.CellYPos = d2{i1}.CellYPosition;
end
%          
end
