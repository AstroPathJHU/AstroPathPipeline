function fltOneBatch(basepath, flatfieldlocation, slideids)
%%
% run the average coding on a list of images
% if the averaging images fail, 
% delete all the mean and csv images
% 
% p1 = fully qualified .bin output image name
% tbl3 = table with sample names, scanpaths, and batchIDs for the
% batch
% fnms = cell array of mean.flt files for averaging
%%
slideids = strsplit(slideids, ',');
f1 = fullfile(basepath, slideids{1}, 'im3\',[slideids{1},'-mean.flt']);
f2 = replace(f1,'.flt','.csv');
nn  = csvread(f2);
k = nn(2);
h = nn(4);
w = nn(3);
%
mean2flat(flatfieldlocation,basepath,slideids,100,k, h, w);
%
end
%
function [B] = mean2flat(fname,basepath,slideids,g,k, m, n)
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
for i1=1:numel(slideids)
    %
    f1 = fullfile(basepath, slideids{i1}, 'im3\',[slideids{i1},'-mean.flt']);
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
for i1=1:k
    B(:,:,i1) = imgaussfilt(A(:,:,i1),g);
    bmean    = mean(B(:,:,i1),'all');
    B(:,:,i1) = B(:,:,i1) ./ bmean;
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