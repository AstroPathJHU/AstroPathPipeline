function [a,b,c] = test(C,n)
    C.tiffext = sprintf('_component_data.tif');
    C.tiffpath = [C.root '\' C.samp '\inform_data\Component_Tiffs\'];
    f = [C.tiffpath, replace(C.R.file{n},'.im3',C.tiffext)];
    g = [C.lumipath,'\',replace(C.R.file{n},'.im3','-lumi.tif')];
    %
    a = imread(f,1);
    %b = imread(g);    
    %c = 3*a.*xform(b)./b;
    c = xform1(a);
    max(c(:))
    imshow(uint8(c));
    shg
end
