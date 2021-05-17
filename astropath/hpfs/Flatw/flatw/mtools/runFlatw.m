function d=runFlatw(path, fpath, sample)
%%------------------------------------------------------
%% Apply the flat field to all images in the directory, and
%% write the flatfielded and warped images back to the same
%% place, in raw format, with .fw extension
%% Alex Szalay, Baltimore, 2018-05-38
%%
%% Usage: runFWloop('F:\new3\M27_1');
%%------------------------------------------------------
    %
    tic;
    %
    [~, ~, BatchID] = getscan(path, sample);
    %
    p1 = [path,'\',sample];
    %
    % load flat and warp. flat is in raw ordering
    %
    fd = fopen([path,'\Flatfield\flatfield_BatchID_',BatchID,'.bin'],'r');
    flat = fread(fd,'double');
    fclose(fd);
    %
    warp = mkwarp();
    %
    % get the image names
    %
    p1 = [fpath,'\',sample];
    p = [p1,'\**\*.raw'];
    d = dir(p);
    %
    % run the parallel loop
    %
    if isempty(gcp('nocreate'))
        numcores = feature('numcores');
        usecores = 4;
        if usecores > numcores
            usecores = numcores;
        end
        evalc('parpool(usecores)');
    end
    %
    parfor i=1:numel(d)
        doflatwarpcore(flat,warp,d(i));
    end
    %
    dt = toc;
    %
    disp(sprintf('      %d files in %f sec',numel(d), dt));
    %
    %poolobj = gcp('nocreate');
    %T = evalc('delete(poolobj)');
    %
end
function doflatwarpcore(flat,warp,dd)
%%----------------------------------------------
%% inner core of the parallel loop
%% Alex Szalay, Baltimore, 2018-06-10
%%----------------------------------------------
    f1 = fullfile(dd.folder,dd.name);
    f2 = replace(f1,'.raw','.fw');
    %
    r = double(im3readraw(f1));
    %
    if(size(r,1)~=size(flat,1))
        fprintf(1,'%d,%d,%s\n',size(flat,1),size(r,1),dd.name);
        return
    end
    %
    r = (r./flat);
    r = permute(reshape(r,35,1344,1004),[3,2,1]);
    r = imwarp(r,warp);
    r = permute(r,[3,2,1]);
    %
    im3writeraw(f2,uint16(r(:)));
    %
end

function d = mkwarp()
%%-------------------------------------------------------
%% create the warp field with the built-in parameters
%% Alex Szalay, Baltimore, 2018-06-10
%%-------------------------------------------------------
    %
    n = 1344;
    m = 1004;
    %
    xc = 584;
    yc = 600;
    wm = 1.85;
    %
    x  = (repmat((1:n) ,m,1)-xc)/500;
    y  = (repmat((1:m)',1,n)-yc)/500;    
    r2 = x.^2 + y.^2;
    r1 = sqrt(r2);
    r3 = r2.*r1;
    %
    % 4-th order polynomial fit, with max warp of pixels
    %
    f = pfit(wm);
    c = (f.p1*r3+f.p2*r2+f.p3*r1+f.p4);        
    %
    d = zeros(m,n,2);
    d(:,:,1) = c.*x;
    d(:,:,2) = c.*y;
    %
end

function fitresult = pfit(a)
%%------------------------------------------------
%% fit a 3rd order polynomial to the warping
%%------------------------------------------------
    %
    x = [0,0.2,0.4,0.8,1.4];
    y = [0,0,0, 0.2,a];
    [xData, yData] = prepareCurveData( x, y );
    %
    % Set up fittype and options
    ft = fittype( 'poly4' );
    %
    % Fit model to data
    [fitresult, gof] = fit( xData, yData, ft );
    %
end

function im3writeraw(fname,a)
%%---------------------------------------------------------------
%% write the binary blob of the body of an IM3 file into a file
%%---------------------------------------------------------------
    %
    fd = fopen(fname,'w');
    ii = 0;
    while fd == -1
        fd = fopen(fname,'w');
        if ii > 6
            disp('Error: could not find ', fname)
            return
        end
        ii = ii + 1;
    end
    fwrite(fd,a(:),'uint16');
    fclose(fd);
    %
end

function aa = im3readraw(fname)
%%---------------------------------------------------------------
%% load the binary dump of the body of an IM3 file into memory
%%---------------------------------------------------------------
    %
    fd = fopen(fname,'r');
    aa = uint16(fread(fd,'uint16'));    
    fclose(fd);
    %
end
%% getscan
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% get the highest number of a directory given a directory and a specimen
%%% name
%% --------------------------------------------------------------
%%
function [Scanpath, ScanNum, BatchID] = getscan(wd, sname)
%
% get highest scan for a sample
%
Scan = dir([wd,'\',sname,'\im3\Scan*']);
%
if isempty(Scan)
    Scanpath = [];
    ScanNum = [];
    BatchID = [];
    return
end
%
sid = {Scan(:).name};
sid = vertcat(sid{:});
sid = sort(sid,1,'ascend');
ScanNum = sid(end);
%
Scanpath = [wd,'\',sname,'\im3\Scan', num2str(ScanNum)];
BatchID  = [];
fid = fullfile(Scanpath, 'BatchID.txt');
try
    fid = fopen(fid, 'r');
    BatchID = fscanf(fid, '%s');
    fclose(fid);
catch
end

if ~isempty(BatchID) && length(BatchID) == 1
    BatchID = ['0',BatchID];
end
Scanpath = [Scanpath,'\'];
end
