function s = copyLowZoom()
    %
    s = readtable('W:\bki\save\samples1.csv');
    %
    zoom = {'1','2','3','4','5'};
    ff = 'e:\zoom\';
    gg = 'Y:\zoom\';
    for n=1:numel(s.SampleID)
        for i = 1:numel(zoom)
            f = [ff,s.SlideID{n},'\',zoom{i}];
            g = [gg,s.SlideID{n},'\',zoom{i}];
            copyfile(f,g);
        end 
    end
end