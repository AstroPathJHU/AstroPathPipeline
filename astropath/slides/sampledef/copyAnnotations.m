function d=copyAnnotations()
    %
    oldpath = '\\bki02\e\Clinical_Specimen';
    newpath = '\\bki04\Clinical_Specimen';
    
    d = dir([oldpath,'\*\im3\Scan*\*.polygons.xml']);
    d=struct2table(d);
    d.newfolder=replace(d.folder,oldpath,newpath);
    %
    for i=1:numel(d.bytes)
       src = fullfile(d.folder{i},d.name{i});
       dst = fullfile(d.newfolder{i},d.name{i});
       fprintf('%s -> %s\n', src, dst);
       copyfile(src,dst);
    end
    %
end