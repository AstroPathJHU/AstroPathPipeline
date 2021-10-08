%% function: getOneSampleSeg; 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
%%% create the component_data_w_seg.tif files in the *\Component_Tiffs
%%% directory. Layers are as follows: 
%%% Layers 1:8 - component data
%%% Layer 9: - Tissue Segmentation
%%% Layer 10-11: Nuclues -- immune; then alternative segmentations
%%% Layer 12-13: Membrane -- immune; then alternative segmentations
%% --------------------------------------------------------------
%%
function [] = GetaSeg(basepath, slideid, MergeConfig)

%
tim = cell(4,1);
%
tim{1} = datestr(now,'dd-mmm-yyyy HH:MM:SS');
%
tic
errors2 = [];
errors = cell(1,1);
%
% get Markers structure
%
try
    [Markers,~] = createmarks(MergeConfig);
catch
    errors2 = ['Error in ',slideid, ': check Merge Config files.'];
    disp(errors2);
    return
end
%
if isempty(gcp('nocreate'))
    try
        numcores = feature('numcores');
        if numcores > 6
            numcores = floor(numcores/4);
        end
        evalc('parpool("local",numcores)');
    catch
        try
            numcores = feature('numcores');
            if numcores > 6
                numcores = floor(numcores/4);
            end
            evalc('parpool("BG1",numcores)');
        catch
        end
    end
end
%
wd1 = [basepath,'\',slideid,'\inform_data\Phenotyped\Results\Tables\*_table.csv'];
%
fnames = dir(wd1);
%
tim{2} = length(fnames);
%
nms = {fnames(:).name};
fds = {fnames(:).folder};
%
% loop through the seg function for each sample with error catching
%
parfor i1 = 1:length(nms)
    %
    wd1 = fullfile(fds{i1},nms{i1});
    a = readtable(wd1);
    %
    fold =  extractBefore(fds{i1},'Phenotyped');
    nam = extractBefore(nms{i1},'_cleaned');
    %
    % print out component_data_w_seg
    %
    try 
        getsegfiles(Markers, a, fold, nam);
    catch
        disp(['Error in ',nms{i1},'.']);
        errors{i1} = nms{i1};
    end
        %
   %disp(nms{i1})
end
%
poolobj = gcp('nocreate');
delete(poolobj);
%
tim{3} = toc;
%
filenms = dir([basepath,'\',slideid,'\inform_data\Phenotyped\Results\Tables\*_cleaned_phenotype_table.csv']);
tim{4} = length(filenms);
%
createlog(errors, errors2, [basepath,'\',slideid], tim);
%
end

%% function: getsegfiles; 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
%%% carry out the getting the segmentation, changing cellids, and printing
%%% images for a single image. Segmentation should be in the lowest numeric
%%% Opal AB inform output folder for each segmentation type inside 
%%% *\inform_data\Phenotyped; 
%%% Component_Tiffs should be in a separate directory at inform_data
%% --------------------------------------------------------------
%%
function []  = getsegfiles(Markers, q, fold, nam)
p = q;
%
% find the rows which do not have a membrane
%
mrows = isnan(p.MeanMembraneDAPI);
%
p = p(~mrows,:);
%
% make global cellids for the image before segmentation correction
%
CellID = p.('CellID');
CellID = double(CellID);
CellNum = p.('CellNum');
CellNum = double(CellNum);
Phenotype = p.('Phenotype');
%
% get segmentation outlines for all alternative segmentations
%
im3 = [];
in3 = [];
%
trows = false(length(CellID),length(Markers.altseg));
%
for i1 = 1:length(Markers.altseg)
    markalt = Markers.altseg{i1};
    %
    % get folder and image names for altseg
    %
    fdname = [fold,'Phenotyped\',markalt];
    iname = fullfile(fdname,[nam,'_binary_seg_maps.tif']);
    %
    % get rows of altseg cells
    %
    trows(:,i1) = strcmp(Phenotype, markalt);
    %
    % get inForm cellids of altseg cells
    %
    tcellnums = CellNum(trows(:,i1));
    tcellids = CellID(trows(:,i1));
    %
    % convert images to a single column vector and change cell nums to
    % cellids
    %
    % Membrane
    %
    im = imread(iname,4);
    %
    im = reshape(im,[],1);
    [a,b] = ismember(im,tcellnums); 
    ii2 = b(a,:);
    ii2 = tcellids(ii2);
    imn = zeros(size(im));
    imn(a,:) = ii2;
    %
    im3(:, i1 + 1) = imn;
    %
    % Nucleus
    %
    in = imread(iname, 2);
    %
    in = reshape(in,[],1);
    [a,b] = ismember(in,tcellnums); 
    ii2 = b(a,:);
    ii2 = tcellids(ii2);
    inn = zeros(size(in));
    inn(a,:) = ii2;
    %
    in3(:,i1 + 1) = inn;
    %
end
%
%get filenames for 1ry seg images
%
fdname = [fold,'Phenotyped\',Markers.seg{1}];
%
iname = fullfile(fdname,[nam,'_binary_seg_maps.tif']);
%
% get cellids of 1ry seg cells
%
trowsall = sum(trows,2) > 0;
cellids = CellID(~trowsall,1);
cellnums = CellNum(~trowsall,1);
%
% Membrane
%
im = imread(iname,4);
%
im = reshape(im,[],1);
[a,b] = ismember(im,cellnums);
ii2 = b(a,:);
ii2 = cellids(ii2);
imn = zeros(size(im));
imn(a,:) = ii2;
%
im3(:, 1) = imn;
%
% Nucleus
%
in = imread(iname, 2);
%
in = reshape(in,[],1);
[a,b] = ismember(in,cellnums);
ii2 = b(a,:);
ii2 = cellids(ii2);
inn = zeros(size(in));
inn(a,:) = ii2;
%
in3(:,1) = inn;
%
% get tissue seg
%
tisseg = imread(iname,1);
%
% get component_data
%
cfd = [fold, 'Component_Tiffs'];
cnm = [nam,'_'];
%
iname = fullfile(cfd, [cnm,'component_data.tif']);
%
props = imfinfo(iname);
for i1 = 1:(length(props) - 1)
    im = imread(iname, i1);
    cim(:,:,i1) = im; 
end
%
% print the images
%
imsize = [props(1).Height, props(1).Width];
ds.ImageLength = props(1).Height;
ds.ImageWidth = props(1).Width;
ds.Photometric = Tiff.Photometric.MinIsBlack;
ds.BitsPerSample   = props(1).BitDepth;
ds.SamplesPerPixel = 1;
ds.SampleFormat = Tiff.SampleFormat.IEEEFP;
ds.RowsPerStrip    = length(props(1).StripByteCounts);
ds.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
ds.Software = 'MATLAB';
ds.Compression = Tiff.Compression.LZW;
%
iname = fullfile(cfd, [cnm,'component_data_w_seg.tif']);
%
ii = Tiff(iname,'w');
for i3 = 1:(length(props) - 1)
    d = cim(:,:,i3);
    ii.setTag(ds);
    write(ii,d);
    writeDirectory(ii)
end
%
% Tissue
%
ii.setTag(ds)
tisseg = single(tisseg);
write(ii,tisseg);
%
% Nucleus
%
for i3 = 1:(size(in3, 2))
    writeDirectory(ii)
    d = reshape(in3(:,i3), imsize);
    d = single(d);
    ii.setTag(ds);
    write(ii,d);
end
%
% Membrane
%
for i3 = 1:(size(im3,2))
    writeDirectory(ii)
    d = reshape(im3(:,i3), imsize);
    d = single(d);
    ii.setTag(ds);
    write(ii,d);
end
%
close(ii)
end
%
%% function: createlog; Create error log for images in segmentation protocol
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/18/2019
%% --------------------------------------------------------------
%% Description
%%% create a log file in the *\Component_Tiffs directory that tracks any failed
%%% fields that were not mentioned in the inform_error logs
%% --------------------------------------------------------------
%%
function createlog(errors, errors2, basepath, tim)
%
% get non-empty cells, ie the names of images that had errors
%
errors = errors(~cellfun('isempty',errors));
%
% create file
%
logf = [basepath,'\inform_data\Component_Tiffs\SegLog.txt'];
%
% create first line of file
%
tim1 = tim{1};
str = ['Generating Segmentation started - ', tim1, '\r\n'];
tim1 = num2str(tim{2});
if ~isempty(tim1)
    str = [str, '     ',tim1,' *cleaned_phenotype_tables detected. \r\n'];
end
%
% add to list of any errors that may have occured
%
if ~isempty(errors2)
    %
    % if the code never made it passed Createmarks the error message is as
    % follows
    %
    str = [str, errors2,'\r\n'];
else
    if ~isempty(errors)
        str = [str,'There was/were an error(s) on the following image(s): \r\n'];
        for i1 = 1: length(errors)
            %
            % write out errors to the file
            %
            fname = extractBefore(errors{i1}, '_cleaned');
            str = [str,'     Problem processing image "',fname,...
                '": Error in Generating segmentation("',fname,...
                '"). \r\n'];
        end
    else
        %
        % if there were no errors output this message in line2 instead
        %
        tim1 = num2str(tim{3});
        tim2 = num2str(tim{4});
        %
        str = [str,'     Successfully generated segmentation for ',tim2,...
            ' image data tables in ',tim1, ' secs. \r\n'];
        %
    end
end
dt = datestr(now,'dd-mmm-yyyy HH:MM:SS');
str = [str,'Generating Segmentation complete - ',dt,'\r\n'];
%
% now write out the string
%
fileID = fopen(logf,'wt');
fprintf(fileID,str);
fclose(fileID);
%
log = str;
end