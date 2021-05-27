function C=getPmts(csv)
    %
    C.csv = csv;
    C.cores   = readtable([csv,'\cores.csv']);
    C.tissues = readtable([csv,'\tissues.csv']);
    C.markers = readtable([csv,'\markers.csv']);
    %
end