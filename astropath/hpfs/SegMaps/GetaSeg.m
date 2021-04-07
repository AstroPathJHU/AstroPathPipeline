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
function [] = GetaSeg(wd, sname, MergeConfig)

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
    errors2 = ['Error in ',sname, ': check Batch ID files.'];
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
wd1 = [wd,'\Results\Tables\*_table.csv'];
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
filenms = dir([wd,'\Results\Tables\*_cleaned_phenotype_table.csv']);
tim{4} = length(filenms);
%
createlog(errors, errors2, wd, tim);
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

%% function: createmarks; Create Markers data structure
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% function takes in a folder location and creates the Markers data
%%% structure
%% --------------------------------------------------------------
%%
function [Markers, err_val] = createmarks(MergeConfig)
%
warning('off','MATLAB:table:ModifiedAndSavedVarnames')
Markers = [];
%
try
    BIDtbl = readtable(MergeConfig);
    B = BIDtbl(:,{'Opal','Target',...
    'TargetType','CoexpressionStatus','SegmentationStatus',...
    'SegmentationHierarchy', 'ImageQA', 'NumberofSegmentations'});
catch
    err_val = 1;
    return
end
%
% check table input variables
%
[B, err_val] = checkTableVars(B);
if ~ismember(err_val, [0,2])
    return
end
%
% start setting up Markers struct
%
Markers.Opals = B.Opal;
Markers.all = B.Target;
%
ii = strcmp('Tumor',B.ImageQA);
Markers.all_original = Markers.all;
%
% change Tumor marker designation to 'Tumor'
%
if sum(ii) == 1
    Markers.all(ii) = {'Tumor'};
    Markers.Tumor{1} = 'Tumor';
elseif sum(ii) > 1
    err_val = 6;
    return
else
     Markers.Tumor{1} = '';
end
%
% get lineage and expression markers
%
LT = strcmp(B.TargetType,'Lineage');
Markers.lin = Markers.all(LT);
%
ET = strcmp(B.TargetType,'Expression');
Markers.expr = Markers.all(ET);
%
% get the markers with multiple segmentations, this will only be a
% capability on expression markers
%
nsegs = B.NumberofSegmentations;
if iscell(nsegs)
    nsegs = cellfun(@(x) str2double(x), nsegs, 'Uni',0);
    nsegs = cell2mat(nsegs);
end
if find(nsegs(~ET) > 1)
    err_val = 7;
    return
end
Markers.nsegs = nsegs;
%
% Set up segmentation status to define number of segmentations and which is
% the primary segmentation
%
SS = B.SegmentationStatus;
Markers.SegStatus = SS;
%
ii = nsegs == 1 & ~ismember(Markers.all,Markers.expr);
SS = SS(ii);
mn = Markers.all(ii);
%
% get number of different segmentations, remove markers with multiple
% segmentations from the contention
%
[~,y,~] = unique(SS);
ii = y(1);
%
Markers.seg = mn(ii);
%
Markers.altseg = cell(length(y)-1,1);
for i1 = 2:length(y)
    ii = y(i1);
    Markers.altseg(i1-1) = mn(ii);
end
%
% get coexpression status for lineage markers
%
CS = B.CoexpressionStatus(LT);
ii = ~strcmp(CS,'NA') | ~strcmp(CS,'NaN');
CS = CS(ii);
%
% track the corresponding target
%
TCS = Markers.lin(ii);
%
% get segmentation heirarchy
%
SH = B.SegmentationHierarchy;
Markers.SegHie = SH(LT);
%
% CS that is not NA in lineage markers; find which coexpressions are
% acceptable
%
Markers.add = [];
sego = [];
for i1 = 1:length(CS)
    %
    % get current target and opal
    %
    T = TCS{i1};
    ii = strcmp(T,Markers.all);
    o = Markers.Opals(ii);
    o = o{1};
    %
    % check them against rest of targets in coexpression
    %
    CStest = CS(~strcmp(TCS,T));
    TCStest = TCS(~strcmp(TCS,T));
    %
    for i2 = 1:length(CStest)
        o1 = CStest{i2};
        T1 = TCStest{i2};
        %
        % if the current target matches one of the targets in the rest 
        %
        if contains(o1,o)
            %
            % if the Markers.add is not empty; are both markers already
            % contained together
            %
            if ~isempty(Markers.add) && sum(contains(Markers.add,T)...
                & contains(Markers.add,T1))
                continue
            else
                track = length(Markers.add) + 1;
                Markers.add{track} = [T1,T];
                ii = strcmp(T1, Markers.lin);
                seg1 = Markers.SegHie(ii);
                ii = strcmp(T, Markers.lin);
                seg2 = Markers.SegHie(ii);
                %
                seg = max([str2double(seg1{1}),str2double(seg2{1})]);
                sego{track} = num2str(seg);
            end
        end
    end
end
%
Markers.SegHie = [Markers.SegHie;sego'];
%
% get coexpression status for expression markers
%
CS = B.CoexpressionStatus(ET);
for i1 = 1:length(CS)
    T = CS{i1};
    T = reshape(T,3,[])';
    [s,~] = size(T);
    x = arrayfun(@(x)contains(Markers.Opals,T(x,:)),1:s,'Uni',0);
    x = horzcat(x{:});
    Markers.Coex{i1} = sum(x,2);
end
%
% reformat for proper dims
%
Markers.Opals = cellfun(@str2double, Markers.Opals, 'Uni',0);
Markers.Opals = cell2mat(Markers.Opals)';
Markers.all = Markers.all';
Markers.all_original = Markers.all_original';
Markers.lin = Markers.lin';
Markers.expr = Markers.expr';
Markers.nsegs = Markers.nsegs';
Markers.seg = Markers.seg';
Markers.altseg = Markers.altseg';
Markers.SegHie = Markers.SegHie';
%
end
%
function [B, err_val] = checkTableVars(B)
%%
% check the table variables to be sure they are in the correct format for
% the code. If they are not convert them.
%%
%
err_val = 0;
%
% check the data type for Opal column
%
if isa(B.Opal,'double')
   %
   % if B.Opal is a 'double' convert to a string 
   %
   tmpopal = num2cell(B.Opal);
   tmpopal = cellfun(@(x) num2str(x), tmpopal, 'Uni', 0);
   ii = strcmp(tmpopal, 'NaN');
   %
   if sum(ii) > 1
      err_val = 2;
      ii = find(ii,1);
   end
   %
   tmpopal(ii) = {'DAPI'};
   ss = size(tmpopal);
   if ss(1) == 1
       B.Opal = tmpopal';
   else
       B.Opal = tmpopal;
   end
end
%
if ~isa(B.Opal, 'cell')
  err_val = 3;
  return
end
%
% check the data type for the coexpression status column
%
if isa(B.CoexpressionStatus,'double')
   %
   % if B.Opal is a 'double' convert to a string 
   %
   tmpCS = num2cell(B.CoexpressionStatus);
   tmpCS = cellfun(@(x) num2str(x), tmpCS, 'Uni', 0);
   %
   for i1 = 1:length(tmpCS)
       tmpCS_n = tmpCS{i1};
       if length(tmpCS_n) > 3
           ii = 3:3:length(tmpCS_n) - 1;
           t(1:length(tmpCS_n)) = char(0);
           t(ii) = ',';
           tmpCS_n = [tmpCS_n;t];
           tmpCS_n = reshape(tmpCS_n(tmpCS_n ~= 0),1,[]);
           tmpCS{i1} = tmpCS_n;
       end
   end
   %
   B.CoexpressionStatus = tmpCS;
   %
end
%
B.CoexpressionStatus = cellfun(@(x) replace(x, ',',''),...
      B.CoexpressionStatus, 'Uni',0);
%
if ~isa(B.Opal, 'cell')
    err_val = 4;
end
%
% remove the DAPI row
%
dr = strcmp(B.Opal, 'DAPI');
if sum(dr) ~= 1
    err_val = 5;
end
B(dr,:) = [];
%
% check the last 3 columns are all set as numeric
%
SS = B.SegmentationStatus;
if iscell(SS)
    %SS = cell2mat(SS);
    B.SegmentationStatus = str2double(SS);
end
%
SH = B.SegmentationHierarchy;
if ~iscell(SS)
    SH = num2str(SH);
    SH = cellstr(SH);
    B.SegmentationHierarchy = SH;
end
%
SS = B.NumberofSegmentations;
if iscell(SS)
    %SS = cell2mat(SS);
    B.NumberofSegmentations = str2double(SS);
end
%
end
%% function: createlog; Create error log for images in segmentation protocol
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/18/2019
%% --------------------------------------------------------------
%% Description
%%% create a log file in the *\Component_Tiffs directory that tracks any failed
%%% fields that were not mentioned in the inform_error logs
%% --------------------------------------------------------------
%%
function createlog(errors, errors2, wd, tim)
%
% get non-empty cells, ie the names of images that had errors
%
errors = errors(~cellfun('isempty',errors));
%
% create file
%
logf1 = extractBefore(wd, 'Phenotyped');
logf = [logf1,'\Component_Tiffs\SegLog.txt'];
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