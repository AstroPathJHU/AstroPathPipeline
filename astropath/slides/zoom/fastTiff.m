function data = fastTiff(filename,fmax)
    %
    warning('off','all') % Suppress all the tiff warnings
    tstack  = Tiff(filename);
    [I,J] = size(tstack.read());
    data = zeros(I,J,8,'uint8');
    scale = 255/fmax;
    data(:,:,1)  = uint8(tstack.read()*scale);
    for n = 2:8
        tstack.nextDirectory()
        data(:,:,n) = uint8(tstack.read()*scale);
    end
    warning('on','all')
end
