%% create_color_images
%% Created by: Benjamin Green
%% ---------------------------------------------
%% Description
% create 4 images for each input; single expression images for markers
% showing only positive colors, single expression images for markers
% showing all colors, image mosiacs for single expression images showing up
% to 25 + and 25 - cells -- both with and without DAPI
%% ---------------------------------------------
%%
function create_color_images(im, imageidout, Image,...
    im_full_color, im_full_color_seg, data, d, im_nodapi,...
    im_dapi_noseg, im_nodapi_noseg, compartment)
%
stypes = {'','_no_seg'};
if strcmp(compartment, 'Nucleus')
    dotcolor = 'green';
else
    dotcolor = 'white';
end
%
% get the data sample
%
data.neg = d(~data.ii,:);
if height(data.neg) > 25
    data.neg =  datasample(data.neg,25,1,'Replace',false);
end
if height(data.pos) > 25
    data.pos = datasample(data.pos,25,1,'Replace',false);
end
data.mos = cat(1,data.pos,data.neg);
%
for i1 = 1:2
    stype = stypes{i1};
    %
    if i1 == 1
        ims = im;
    else
        ims = im_dapi_noseg;
    end
    %
    Image.image = insertMarker(ims, data.xy,'+','color',dotcolor,'size',1);
    iname = [imageidout,'single_color_expression_image',stype,'.tif'];
    write_image(iname,Image.image,Image)
    %
    % Create the Image Mosiacs for local positive images with dapi
    %
    
    Image.x = data.mos.CellXPos;
    Image.y = data.mos.CellYPos;
    Image.imname = [imageidout,'cell_stamp_mosiacs_pos_neg',stype,'.tif'];
    makemosaics(Image)
    %
    if i1 == 1
        ims = im_nodapi;
    else
        ims = im_nodapi_noseg;
    end
    %
    % Create the Image Mosiacs for local positive images without dapi
    %
    Image.image = insertMarker(ims,data.xy,'+','color',dotcolor,'size',1);
    Image.imname = [imageidout,'cell_stamp_mosiacs_pos_neg_no_dapi',stype,'.tif'];
    makemosaics(Image)
    %
end
%
% create full color image for phenotyped image with dapi
%
imp = insertMarker(im_full_color_seg, data.xy, '+','color','white','size',1);
iname = [imageidout,'full_color_expression_image.tif'];
write_image(iname,imp,Image)
%
imp = insertMarker(im_full_color, data.xy, '+','color','white','size',1);
iname = [imageidout,'full_color_expression_image_no_seg.tif'];
write_image(iname,imp,Image)
%
end
