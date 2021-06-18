function aa = readflat(fname,m,n,k)
%%------------------------------------------
%% read a flatfield file stored as doubles
%% in the original im3 ordering (pixelwise)
%% and convert it to layer-wise order
%%------------------------------------------
%
try
    %fprintf('%s\n',replace(fname,'\','/'));
    fprintf('%s\n',fname);
    fd = fopen(fname,'r');
    aa = fread(fd,'double');
    fclose(fd);
catch
    fprintf('File %s not found\n',fname);
end
%
aa = reshape(aa,k,n,m);
aa = permute(aa,[3,2,1]);
%
end