function copyGeomCellsAll(root1)
    %
    d = readtable('\\bki02\c\BKI\save\samples.csv');
    n = numel(d.SampleID);
    %
    for i=1:n
        samp = d.SlideID{i};
        fprintf('%s\n',samp);
        copyGeomCells(root1, samp);
    end
    %
end


function copyGeomCells(root1,samp)
%%------------------------------------------
%%
%%------------------------------------------
    %
    src  = ['F:\geom\',samp,'\'];
    dest = [root1, '\',samp,'\geom\'];
    %{
    src
    dest
    return
    %}
    if (exist(dest)==0)
        mkdir(dest);
    end
    %
    try
        copyfile(src,dest);
    catch
        fprintf('%s => %s copy failed\n',src, dest);
    end
    %
end
