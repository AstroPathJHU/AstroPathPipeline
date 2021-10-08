%% function: getnoseg;
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
function [] = GetnoSeg(basepath, slideid, MergeConfig)

[Markers,~] = createmarks(MergeConfig);
%
fnd = dir([basepath,'\',slideid,'\inform_data\Component_Tiffs\*component_data.tif']);
fnd1 = dir([basepath,'\',slideid,'\inform_data\Component_Tiffs\*component_data_w_seg.tif']);
nm1 = {fnd(:).name};
nm2 = {fnd1(:).name};
%
nm1 = replace(nm1, '.tif', '');
nm2 = replace(nm2, '_w_seg.tif', '');
%
ii = ~ismember(nm1, nm2);
fna = fnd(ii);
%
if isempty(fna)
    return
end
%
l = 3 + 2*length(Markers.altseg);
%
for i1= 1:length(fna)
    %
    nm = fullfile(fna(i1).folder, fna(i1).name);
    %
    cim = [];
    props = imfinfo(nm);
    for i2 = 1:(length(props) - 1)
        im = imread(nm, i2);
        cim(:,:,i2) = im;
    end
    %
    s= size(cim);
    %
    for i2 = 1:l
        cim(:,:,end+1) = zeros(s(1), s(2), 1);
    end
    %
    s= size(cim);
    %
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
    iname = replace(nm, '.tif','_w_seg.tif');
    %
    ii = Tiff(iname,'w');
    %
    d = single(cim(:,:,1));
    ii.setTag(ds);
    write(ii,d);
    %
    for i3 = 2:(s(3))
        writeDirectory(ii)
        d = single(cim(:,:,i3));
        ii.setTag(ds);
        write(ii,d);
    end
    %
    close(ii)
    %
end