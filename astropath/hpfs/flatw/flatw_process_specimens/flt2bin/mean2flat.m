function [B] = mean2flat(fname,d,g,k, m, n)
%%---------------------------------------------
%% read all the sample mean images in d
%% and generate a 35 layer flat field image with
%% smoothing guassian of size g put result mean
%% image in 'fpath', images should
%% have k layers
%%
%%      mean2flat('Y:\raw', dir('Y:\raw'), 100, 35);
%%
%% Alex Szalay, 2019-04-18
%%---------------------------------------------
%
% d = dir([fpath,'\*.flt']);
% m = 1004;
% n = 1344;
%%---------------------------------------
% loop through all the available samples
%%---------------------------------------
A = zeros(m,n,k);
N = 0;
for i=1:numel(d)
    %
    f1 = [d(i).folder,'\',d(i).name];
    f2 = replace(f1,'.flt','.csv');
    %
    nn  = csvread(f2);
    if nn(1) >= 300
        %fprintf('%s : %d\n', d(i).name, nn);
        A = A + (readflat(f1, m, n, k) .* nn(1));
        N = N + nn(1);
    end
    %
end
A = A ./ N;
%---------------------------------------------
% smooth the images and normalize to mean=1
%---------------------------------------------
fprintf('%d fields averaged, smooth/normalize...\n',N);
for i=1:k
    B(:,:,i) = imgaussfilt(A(:,:,i),g);
    bmean    = mean(B(:,:,i),'all');
    B(:,:,i) = B(:,:,i) ./ bmean;
end
%
if N < 3000
    fprintf(' fields averaged less than 3000 manual interaction needed...\n')
    fprintf(' bin file not generated for batch \n')
    return
end
%
%-----------------------
%
fd = fopen(fname,'w');
C  = permute(B,[3,2,1]);
try
    fwrite(fd,C(:),'double');
catch
end
clear C
fclose(fd);
fname = replace(fname,'bin','csv');
fd = fopen(fname,'w');
try
    fwrite(fd,N,'double');
catch
end
%
fclose(fd);
%
end
%