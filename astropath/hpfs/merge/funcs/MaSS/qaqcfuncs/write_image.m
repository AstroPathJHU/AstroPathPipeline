%% write_image
%% Created by: Benjamin Green
%% ---------------------------------------------
%% Description
% write out the image 'Image' to the file iname using the Tiff library
%% ---------------------------------------------
%
%%
function write_image(iname,im,Image)
T = Tiff(iname,'w');
T.setTag(Image.ds);
write(T,im);
writeDirectory(T)
close(T)
end
