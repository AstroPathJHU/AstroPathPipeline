%%
%
%% ----------------------------------------------------------
% Get the sample names from a Clinical Specimen folder and shred the im3's
% to the flatw folder. These functions will produce the .raw and the .imm
% files which can be read in easily as a single column vector of the image
% files
%
%% -------------------------------------------------------------
%
function [] = process_raw(main, tn, dd)
% wd = '\\bki04\k$\Clinical_Specimen_4';
% flatw = '\\bki04\m$\flatwtest_4\raw';
tbl = readtable([main, '\Paths', tn, '.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
%
for i1 = 1:height(tbl)
    %
    % Clinical_Specimen folder
    %
    wd = tbl(i1,'Main_Path');
    wd = table2array(wd);
    wd = wd{1};
    %
    machinename = tbl(i1,'Machine');
    machinename = table2array(machinename);
    machinename = machinename{1};
    %
    flatwcode = [dd,'\Processing_Specimens\flatwCodes\',machinename];
    flatw = [dd,'\Processing_Specimens\raw'];
    if ~exist(flatw,'dir')
        mkdir(flatw)
    end
    %
    sp = dir(wd);
    sp = sp(3:end);
    ii = [sp.isdir];
    sp = sp(ii);
    sn = {sp(:).name};
    ii = (contains(sn, 'Batch')...
        |contains(sn, 'tmp_inform_data')|...
        contains(sn, 'reject')|...
        contains(sn, 'Control')|...
        strcmp(sn, 'Clinical')|...
        strcmp(sn, 'Upkeep and Progress')|...
        strcmp(sn, 'Flatfield'));
    sn(ii) = [];
    %
    %
    for i2 = 1:length(sn)
        %
        sn1 = sn{i2};
        %
        sp2 = dir([wd,'\',sn1,'\im3\**\*.qptiff']);
        if isempty(sp2)
           % fprintf(['"',sn1,'" is not a valid clinical specimen folder \n']);
            continue
        end
        %
        sp2 = dir([wd,'\',sn1,'\im3\*.flt']);
        if ~isempty(sp2)
            continue
        end
        ShredIm3(wd, flatw, sn1, flatwcode);
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
            rmdir(flatw, 's')
        catch
        end
    end
end
end
%
function [] = ShredIm3(wd, flatw, sample, flatwcode)
%
try
    AScode = [flatwcode,'\Im3Tools\fixM2'];
    system([AScode, ' ', wd,' ', sample,' ',flatwcode]);
    AScode = [flatwcode,'\Im3Tools\shredPath'];
    system([AScode, ' ', wd, ' ', flatw, ' ', sample,' ',flatwcode]);
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
pp = [rawpath, '\', samp, '\*.raw'];
dd = dir(pp);
N  = numel(dd);
%
if (N==0)
    fprintf('Sample %s is not found in %s\n',samp, rawpath);
    return
end
%
if (exist([rawpath,'\flat'])==0)
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
