%% mkimageid function
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% creates variables for a single image
%% --------------------------------------------------------------
%%
function [q, imageida, mycol, imc, simage] =...
    mkimageid(charts, inum, wd, Markers, doseg)
%
% set image output properties
%
imageida.ds.Photometric = Tiff.Photometric.RGB;
imageida.ds.BitsPerSample   = 8;
imageida.ds.SamplesPerPixel = 3;
imageida.ds.SampleFormat = Tiff.SampleFormat.UInt;
imageida.ds.RowsPerStrip = 41;
imageida.ds.MaxSampleValue = 256;
imageida.ds.MinSampleValue = 0;
imageida.ds.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
imageida.ds.Software = 'MATLAB';
imageida.ds.ResolutionUnit = Tiff.ResolutionUnit.Inch;
imageida.ds.XResolution = 300;
imageida.ds.YResolution = 300;
imageida.ds.Compression = Tiff.Compression.LZW;
%
% get chart that correspond to inum
%
nc = [charts(inum).folder,'\',charts(inum).name];
q = load(nc);
q = q.fData;
q.fname = charts(inum);
q.fig.CellXPos = q.fig.CellXPos + 1;
q.fig.CellYPos = q.fig.CellYPos + 1;
%
% some image designations
%
imageida.wd = wd;
imageida.id = extractBefore(q.fname.name,'cleaned_phenotype_table.mat');
%
% write out Tables that comes from this image
%
writetable(q.fig,[wd,'\Phenotyped\Results\QA_QC\Tables_QA_QC\',...
    erase(q.fname.name,'.mat'),'.csv']);
%
% image input fname for segmentation images
%
sim{1} = [wd,'\Phenotyped\',Markers.seg{1},'\',imageida.id];
for i1 = 1: length(Markers.altseg)
    sim{i1+1} = [wd,'\Phenotyped\',Markers.altseg{i1},'\',imageida.id];
end
%
% image output fname for the full Marker images
%
imageida.outfull = [wd,...
    '\Phenotyped\Results\QA_QC\Phenotype\All_Markers\',imageida.id];
%
% image output fname for lineage markers
%
for i1 = 1:length(Markers.lin)
    imageida.outABlin{i1} = [wd,...
        '\Phenotyped\Results\QA_QC\Phenotype\',Markers.lin{i1},'\',imageida.id];
    imageida.outABcoex{i1} = [wd,'\Phenotyped\Results\QA_QC\Lin&Expr_Coex\',...
        Markers.lin{i1},'\',imageida.id];
end
%
% image output fname name for additional lineage markers (ie coexpression)
% image output fname for expression marker coexpression on lineage markers
%
for i2 = 1:length(Markers.add)
    imageida.outABlin{i1+1} = [wd,'\Phenotyped\Results\QA_QC\Phenotype\',...
        Markers.add{i2},'\',imageida.id];
    imageida.outABcoex{i1+1} = [wd,'\Phenotyped\Results\QA_QC\Lin&Expr_Coex\',...
        Markers.add{i2},'\',imageida.id];
    i1 = i1+1;
end
%
% image output fname for expression markers
%
for i1 = 1: length(Markers.expr)
    imageida.outABexpr{i1} = [wd,'\Phenotyped\Results\QA_QC\Phenotype\',...
        Markers.expr{i1},'\',imageida.id];    
end
ii = ismember(Markers.all, Markers.expr);
imageida.exprlayer = Markers.Opals(ii);
%
idx = find(Markers.nsegs > 1);
idx_count = length(imageida.outABexpr);
%
if idx
    for i1 = 1:length(idx)
        cidx = idx(i1);
        for i2 = 2:Markers.nsegs(cidx)
            idx_count = idx_count + 1;
            str = [wd,'\Phenotyped\Results\QA_QC\Phenotype\',...
                Markers.all{cidx},'_',num2str(i2)];
            if ~exist(str, 'dir')
                mkdir(str);
            end
            imageida.outABexpr{idx_count} = [str,'\',imageida.id];
            imageida.exprlayer = [imageida.exprlayer;Markers.Opals(cidx)];
        end
    end
end
%
% fname for the component_Tiff image
%
iname = [wd,'\Component_Tiffs\',...
    imageida.id,'component_data.tif'];
%
% read in all component images
%
props = imfinfo(iname);
imageida.size = [props(1).Height, props(1).Width];
%
imageida.ds.ImageLength = props(1).Height;
imageida.ds.ImageWidth = props(1).Width;
ii = cellfun(@(x) strcmp(x, 'grayscale'), {props.ColorType});
layers = sum(ii);
%
for i2 = 1:layers
    if strcmp(props(i2).ColorType, 'grayscale')
        im(:,1) = reshape(imread(iname,i2),[],1);
        imc(:,i2) =(im(:,1)./max(im(:,1)));
    end
end
%
mycol.all = Markers.mycol.all;
%
% lineage marker colors only
%
lins = ismember(Markers.all,Markers.lin);
mycol.lin = mycol.all(2:end-1,:);
mycol.lin = mycol.lin(lins,:);
%
% expression marker colors only
%
expr = ismember(Markers.all,Markers.expr);
mycol.expr = mycol.all(2:end-1,:);
mycol.expr = mycol.expr(expr,:);
%
%%%segmentation images%%%
%
if doseg
    %
    % get rows from each alternative segmentation in the main table
    %
    trows = false(height(q.fig),length(Markers.altseg));
    for i1 = 1:length(Markers.altseg)
        trows(:,i1) = strcmp(q.fig.Phenotype,Markers.altseg{i1});
        cellnums = double(q.fig.CellNum(trows(:,i1)));
        %
        % read in alternative segmentations; this only works if there is 
        % tissue segmentation and nuclear segmentation in the 
        % binary_seg image; cytoplasm
        %
        s1 = imread([sim{i1 + 1},'binary_seg_maps.tif'], 4);
        %
        % set cell labels of segmentation image that are not 
        % in the main table to zero
        %
        s1(~ismember(double(s1),cellnums)) = 0;
        %
        s1 = reshape(s1,[],1);
        simage3{i1 + 1} = s1;
    end
    %
    % get every row for alternative segmentation in the main table
    %
    trowsall = sum(trows,2) > 0;
    %
    % read in primary segmentation image
    %
    s1 = imread([sim{1},'binary_seg_maps.tif'],4);
    %
    % get cellnums of primary segmentation data
    % (ie data not in any alt segs)
    %
    cellnums = double(q.fig.CellNum(~trowsall,:));
    %
    s1(~ismember(double(s1),cellnums))=0;
    s1 = reshape(s1,[],1);
    %
    simage3{1} = s1;
    %
    % read in tissue segmentation
    %
    % sum the images across the segmentations to create a single unique
    % segmentation
    %
    simage2 = [simage3{:}];
    %
    simage = sum(simage2,2);
    %
    simage(simage>0) = .5;
    %
    simage = reshape(double(simage), imageida.size);
else
    simage = zeros(imageida.size);
end
%
end
