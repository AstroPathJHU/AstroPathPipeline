function tiffwrite(timg, fname)
%%-----------------------------------------
%% write img into a grayscale tiff file 
%% using single precision floats
%%-----------------------------------------
    %
    %info = imfinfo(fname);
    t = Tiff(fname, 'w');
    tagstruct.ImageLength = size(timg, 1);
    tagstruct.ImageWidth  = size(timg, 2);
    tagstruct.Compression = Tiff.Compression.None;
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct2.Software = 'MATLAB';
    t.setTag(tagstruct);
    t.write(timg);
    t.close();
    %
end
