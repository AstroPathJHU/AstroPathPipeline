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
function [] = meanimages(main, dd)
% wd = '\\bki04\k$\Clinical_Specimen_4';
% flatw = '\\bki04\m$\flatwtest_4\raw';
tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
%
% set up worker
%
filepath = fileparts(mfilename('fullpath'));
flatwcode = [filepath, '\..\Flatfield\flatw'];
if ~exist(flatwcode, 'dir')
    disp('ERROR: raw2mean_loop worker not set up')
    return
end
%
for i1 = 1:height(tbl)
    %
    % Clinical_Specimen folder
    %
    wd = tbl(i1,:);
    wd = ['\\', wd.Dpath{1},'\', wd.Dname{1}];
    %
    sn = find_specimens(wd);
    %
    % for each specimen check for .flt, if none create otherwise skip
    %
    for i2 = 1:length(sn)
        sn1 = sn{i2};
        sample_loop(wd, dd, sn1, flatwcode);
    end
end
end
%
function [] = sample_loop(wd, dd, sn1, flatwcode)
        %
        flatw = [dd,'\Processing_Specimens\raw'];
        if ~exist(flatw,'dir')
            mkdir(flatw)
        end
        %
        sp2 = dir([wd,'\',sn1,'\im3\**\*.qptiff']);
        if isempty(sp2)
            return
        end
        %
        sp2 = dir([wd,'\',sn1,'\im3\*.flt']);
        if ~isempty(sp2)
            return
        end
        %
        ShredIm3(wd, flatw, sn1, flatwcode);
        %
        try
            fn = raw2mean(flatw,sn1);
            copyfile(fn,[wd,'\',sn1,'\im3']);
            fn = replace(fn, '.csv','.flt');
            copyfile(fn,[wd,'\',sn1,'\im3']);
        catch
            fprintf(['Error in ',sn1,'\r'])
        end
        fclose('all');
        try
           % rmdir(flatw, 's')
        catch
        end
end
%
function [] = ShredIm3(wd, flatw, sample, flatwcode)
%
try
    AScode = [flatwcode,'\Im3Tools\fixM2'];
    system([AScode, ' ', wd,' ', sample,' ',flatwcode]);
    AScode = [flatwcode,'\Im3Tools\ConvertIm3Path.ps1'];
    shredstring = 'PowerShell -NoProfile -ExecutionPolicy Bypass -Command "& ';
    system(strcat(shredstring, " '", AScode, "' '", wd, "' '", flatw, "' '", sample,"'",'" -s'));
catch
end
end
%
function fn = raw2mean(rawpath, samp)
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
pp = [rawpath, '\', samp, '\*.dat'];
dd = dir(pp);
N  = numel(dd);
%
if (N==0)
    fprintf('Sample %s is not found in %s\n',samp, rawpath);
    return
end
%
if ~exist([rawpath,'\flat'], 'dir')
    mkdir([rawpath,'\flat']);
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
[ll, ww, hh] = get_shape([rawpath,'\',samp], samp);
N = [N,ll,ww,hh];
%
%size(aa)
fn = [rawpath,'\flat\',samp,'-mean.flt'];
%
fd = fopen(fn,'w');
fwrite(fd,aa,'double');
fclose(fd);
%
fn = [rawpath '\flat\',samp,'-mean.csv'];
csvwrite(fn,N);
%
end
%
function sn = find_specimens(wd)
sp = dir(wd);
sp = sp(3:end);
ii = [sp.isdir];
sp = sp(ii);
sn = {sp(:).name};
ii = (contains(sn, 'Batch')|...
    strcmp(sn, 'Clinical')|...
    contains(sn, 'Control')|...
    strcmpi(sn, 'Ctrl')|...
    strcmpi(sn, 'dbload')|...
    strcmpi(sn, 'Flatfield')|...
    strcmpi(sn, 'logfiles')|...
    strcmpi(sn, 'reject')|...
    contains(sn, 'tmp_inform_data')|...
    strcmp(sn, 'Upkeep and Progress')|...
    strcmpi(sn, 'upkeep_and_progress'));
sn(ii) = [];
end

function [ll, ww, hh] = get_shape(path, sample)
%%------------------------------------------------------
%% get the shape parameters from the parameters.xml file found at
%% [path/sample.Parameters.xml]
%% Benjamin Green, Baltimore, 2021-04-06
%%
%% Usage: get_shape('F:\new3\M27_1\im3\xml', 'M27_1');
%%------------------------------------------------------
    p1 = fullfile(path, [sample,'.Parameters.xml']);
    mlStruct = parseXML(p1);
    params = strsplit(mlStruct(5).Children(2).Children.Data);
    ww = str2double(params{1});
    hh = str2double(params{2});
    ll = str2double(params{3});
end