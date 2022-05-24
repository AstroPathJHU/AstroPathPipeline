%% makemosaics
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%%  make the cell stamp mosiac images
%% --------------------------------------------------------------
%%
function makemosaics(Image)
%%
%%%%%This function makes mosiacs to compare two ABs%%%%%%%%%%%%%%%%%%%%%%%%%
%
%image is a data structure with at least 5 fields
%.size - pixel size of cut outs
%.x & .y - x&y coordinates of cell/mosaic centers
%.image - the actual image to make the mosiac
%.imname - the path and file name
%
if ~isempty(Image.x)
    %
    %cut the image
    %
    xmins = Image.x - Image.mossize/2;
    ymins = Image.y - Image.mossize/2;
    h = repmat(Image.mossize, length(xmins),1);
    w = repmat(Image.mossize, length(xmins),1);
    box = [xmins ymins h w];
    mos = cell(length(xmins),1);
    %
    % find dimensions of the grid and number of blank fields
    %
    N = length(xmins);
    c = floor(sqrt(N));
    r = ceil(N/c);
    dims = (c * r);
    blnk = dims - N;
    blnk = blnk + N;
    %
    % get image cut outs
    %
    for i1 = 1:N
        m = imcrop(Image.image,box(i1,:));
        mos{i1,1} = imresize(m,...
            [Image.mossize + 1 Image.mossize + 1]);
    end
    %
    % add blank images
    %
    bb = zeros(Image.mossize + 1,Image.mossize + 1,3, 'uint8');
    for i1 = (N+1):blnk
        mos{i1,1} = bb;
    end
    %
    % make the mosaic and figure
    %
    mos = cell2mat(reshape(mos(:,:,:), r, c).');
    %
    rim = 2.5 * r * Image.mossize + 1;
    cim = 2.5 * c * Image.mossize + 1;
    %
    mos1 = imresize(mos,[cim, rim]);
    Image.ds.ImageLength = cim;
    Image.ds.ImageWidth = rim;
    %
    % print image
    %
    T = Tiff(Image.imname,'w');
    T.setTag(Image.ds);
    write(T,mos1);
    writeDirectory(T)
    close(T)
end
end
