%% launch_meanimages
%% Created by: Benjamin Green - Johns Hopkins 2020-04-06
%% ----------------------------------------------------------
% using the paths table in <main>, loop through each project.
% check each sample for the mean flatfield image, if it does
% not exist create it.
%% Input
% main[string]: name of the directory with the paths document
% dd[string]: a drive set up with the astropathpipeline code for processing
% this should include the entire repository filed under a
% 'Processing_Specimens' folder.
%% Usage
% raw2mean_loop('\\bki04\astropath_processing','\\bki08\e$')
%% -------------------------------------------------------------
%
function fn = raw2mean(fwpath, samp)
%%------------------------------------------------------
%% read all raw im3 images in a given path and
%% calculate the 35-band mean without perfroming
%% the reordering. writes two files, one containing
%% the mean image block, the other is a csv with N.
%%
%%      raw2mean('Y:\raw','M24_1');
%%
%% Alex Szalay, 2019-04-18
%%------------------------------------------------------
%
pp = [fwpath, '\', samp, '\*.dat'];
dd = dir(pp);
N  = numel(dd);
%
if (N==0)
    fprintf('Sample %s is not found in %s\n',samp, fwpath);
    return
end
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
[ll, ww, hh] = get_shape([fwpath,'\',samp], samp);
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