%% process_flt2bin
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 05/14/2019
%% --------------------------------------------------------------
%% Description:
%%% for a batch of specimens in a directory; check the status of the
%%% flatwarping .bin image. if it is not already created, make it
%% --------------------------------------------------------------
%%
function process_flt2bin(main)
%
tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
%
for i1 = 1:height(tbl)
    %
    % Clinical_Specimen folder
    %
    wd = tbl(i1,:);
    wd = ['\\', wd.Dpath{1},'\', wd.Dname{1}];
    %
    samplenames = find_specimens(wd);
    %
    % get scan path and batchID of each sample
    %
    tbl2 = getSampleTable(wd, samplenames);
    %
    % get unique BatchIDs and cycle through to see if .flt file exists
    % for that batch. If it does not check for the mean flat field 
    % files of each specimen
    %
    Batches = unique(tbl2.BatchID);
    %
    if ~exist([wd,'\flatfield'],'dir')
        mkdir([wd,'\flatfield'])
    end
    %
    for i2 = 1:length(Batches)
        B1 = Batches{i2};
        %
        % check if the final .flt file exists for the batch
        %
        p1 = [wd,'\flatfield\flatfield_BatchID_',B1,'.bin'];
        if exist(p1,'file')
            continue
        end
        [ii2, tbl3, fnms] = checkMean(wd, B1, tbl2);
        %
        %
        f = dir([wd,'\upkeep_and_progress\AstropathAPIDdef_*']);
        f = fullfile(f(1).folder,f(1).name);
        tbl4 = readtable(f);
        %
        ii4 = tbl4.BatchID == str2double(B1);
        %
        % if all mean.flt files exist for every specimen then create a total
        % .bin flat field file
        %
        if sum(ii2)==0 && length(ii2) == sum(ii4)
            %
            fltOneBatch(tbl3, p1, fnms)
            %
        end
        %
    end
    %
end
%
end
%
function tbl2 = getSampleTable(wd, samplenames)
%%
% get the table with the scanpath and batchid
% for each sample
%
%%
tbl2 =  cell2table(cell(0,3), 'VariableNames',...
    {'Sample','BatchID','Scanpath'});
%
for i2 = 1:length(samplenames)
    sname = samplenames{i2};
    %
    try
        [Scanpath, ~, BatchID] = getscan(wd, sname);
    catch
        fprintf(['"',sname,'" is not a valid clinical specimen folder \n']);
        continue
    end
    %
    if isempty(BatchID)
        fprintf(['"',sname,'" is not a valid clinical specimen folder \n']);
        continue
    end
    %
    tbl3 = table();
    tbl3.Sample = {sname};
    tbl3.BatchID = {BatchID};
    tbl3.Scanpath = {Scanpath};
    tbl2 = [tbl2;tbl3];
end
%
end
%
function [ii2, tbl3, fnms] = checkMean(wd, B1, tbl2)
%%
% check if the mean flat field file for each sample exists
% exclude batches with, 'artifact_detected.csv' in the im3 path
%
% B1 = string, current batch number 
% wd = string, current main directory
% tbl2 = table of samples, batchids, and scanpaths for all samplesi
% in wd
%
%%
%
% check if the other .flt files exist
%
ii = strcmp(tbl2.BatchID,B1);
tbl3 = tbl2(ii,:);
%
p = cellfun(@(x)[wd,'\',x,'\im3\*artifact_detected.csv'],tbl3.Sample,'Uni',0);
fnms = cellfun(@(x)dir(x),p,'Uni',0);
ii2 = cellfun(@(x)~isempty(x),fnms,'Uni',0);
ii2 = [ii2{:}];
tbl3(ii2,:) = [];
%
p = cellfun(@(x)[wd,'\',x,'\im3\*mean.flt'],tbl3.Sample,'Uni',0);
fnms = cellfun(@(x)dir(x),p,'Uni',0);
ii2 = cellfun(@(x)isempty(x),fnms,'Uni',0);
ii2 = [ii2{:}];
%
end
%
function fltOneBatch(tbl3, p1, fnms)
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
onms = p1;
%
f1 = fullfile(fnms{1}.folder,fnms{1}.name);
f2 = replace(f1,'.flt','.csv');
nn  = csvread(f2);
k = nn(2);
h = nn(4);
w = nn(3);
%
try
    mean2flat(onms,[fnms{:}],100,k, h, w);
catch
    nm = [fnms{:}];
    nm = strcat({nm(:).folder},'\',{nm(:).name});
    delete(nm{:})
    nm = replace(nm,'.flt','.csv');
    delete(nm{:})
end
end
%
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