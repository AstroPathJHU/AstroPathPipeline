%% raw2mean
%% Created by: Benjamin Green and Alex Szalay - Johns Hopkins 2020-04-06
%% ----------------------------------------------------------
%% read all raw im3 images in a given path and
%% calculate the 35-band mean without perfroming
%% the reordering. writes two files, one containing
%% the mean image block, the other is a csv with N, height, width, n layers
%% Input
% fwpath[string]: path to the raw path folder for the sample
% samp[string]: sample name
%% Usage
% raw2mean('Y:\raw','M24_1')
%% -------------------------------------------------------------
%
function fn = raw2mean(fwpath, samp)
%
pp = [fwpath, '\', samp, '\*.dat'];
dd = dir(pp);
N  = numel(dd);
%
if (N==0)
    error('Sample %s is not found in %s',samp, fwpath);
end
%
[ll, ww, hh] = get_shape([fwpath,'\',samp], samp);
%
if ~exist([fwpath,'\flat\', samp], 'dir')
    mkdir([fwpath,'\flat\', samp]);
end
%--------------------------------
% read and stack the raw images
%--------------------------------
fn = [dd(1).folder, '\', dd(1).name];
fd = fopen(fn,'r');
aa = double(fread(fd,'uint16'));
fclose(fd);
%
for n=2:N
    fn = [dd(n).folder, '\', dd(n).name];
    fd = fopen(fn,'r');
    aa = aa + double(fread(fd,'uint16'));
    fclose(fd);
end
aa = aa/N;
%
N = [N,ll,ww,hh];
%
%size(aa)
fn = [fwpath,'\flat\',samp,'\',samp,'-mean.flt'];
%
fd = fopen(fn,'w');
fwrite(fd,aa,'double');
fclose(fd);
%
fn = [fwpath '\flat\',samp,'\',samp,'-mean.csv'];
csvwrite(fn,N);
%
end
%
function [ll, ww, hh] = get_shape(mpath, sample)
%%------------------------------------------------------
%% get the shape parameters from the parameters.xml file found at
%% [path/sample.Parameters.xml]
%% Benjamin Green, Baltimore, 2021-04-06
%%
%% Usage: get_shape('F:\new3\M27_1\im3\xml', 'M27_1');
%%------------------------------------------------------
    p1 = fullfile(mpath, [sample,'.Parameters.xml']);
    mlStruct = parseXML(p1);
    params = strsplit(mlStruct(5).Children(2).Children.Data);
    ww = str2double(params{1});
    hh = str2double(params{2});
    ll = str2double(params{3});
end