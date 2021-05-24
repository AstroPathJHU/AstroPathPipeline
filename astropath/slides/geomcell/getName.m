function f = getName(C,n)
    f = [C.root,'\',C.samp,'\inform_data\Component_Tiffs\'];
    f = [f, replace(C.H.file{n},'.im3','_component_data_w_seg.tif')];
end