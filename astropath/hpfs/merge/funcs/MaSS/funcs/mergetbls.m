function [fData, e_code] = mergetbls(fname, Markers, wd, imall)
%
fData = [];
%
% read in data
%
[C, units, e_code] = readalltxt(fname, Markers, wd);
%
if e_code ~= 0
    return
end
%
% units check
%
if any(~strcmp(units, 'pixels'))
    e_code = 17;
end
%
% select proper phenotypes
%
f = getphenofield(C, Markers, units);
%
% remove cells within X number of pixels in cells
%
d = getdistinct(f, Markers);
%
% removes double calls in hierarchical style
%
q = getcoex(d, Markers);
%
% get polygons from inform and remove other cells inside secondary 
% segmentation polygons
%
a = getseg(q, Markers);
%
% determine expression markers by cells that are in X radius
%
fData = getexprmark(a, Markers);
%
% save the data
%
ii = parsave(fData, fname, Markers, wd, imall);
%
end
%% function: readalltxt; read all table for a given fname
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description:
% function takes in a filename, a data structure with marker information,
% and a directory location
% it loads the *cell_seg_data.txt for each marker output with the filename
% in the file tree described below in the input section
%% --------------------------------------------------------
%%
function [C, units, e_code] = readalltxt(filename,Markers,wd)
%%-------------------------------------------------------------------------
%% get all the phenotypes, coordinates, intensties into a single table
%% 
%%-------------------------------------------------------------------------
e_code = 0;
%
% Create a string vector which contains desired variables from the inform
% output and new variable names
%
vars.Opals = Markers.Opals;
layers = length(Markers.Opals) + 2;
vars.names = cell(1,length(Markers.all)*8);
vars.type = [repmat({'Mean'},1,4), repmat({'Total'},1,4)];
vars.comp = repmat({'Nucleus', 'Membrane','EntireCell','Cytoplasm'},1,2);
%
% create DAPI column names -- there are additional vectors a2 - a4 are
% created so that the program is capable of handling different inform
% outputs
%
a(:,1)= cellfun(@(x,y)strcat(x,'DAPI','_DAPI_',...
    y,'_NormalizedCounts_TotalWeighting_'),vars.comp,vars.type,'Uni',0);
a2(:,1)= a(:,1);
%
% Sometimes DAPI is named, sometimes it isn't resulting in different
% column names which we use these vectors
%
a3(:,1)= cellfun(@(x,y)strcat(x,'DAPI',...
    y,'_NormalizedCounts_TotalWeighting_'),vars.comp,vars.type,'Uni',0);
a4(:,1)= a3(:,1);
%
b(:,1)= cellfun(@(x,y)strcat(y,x,'DAPI'),vars.comp,vars.type,'Uni',0);
%
% column names for all other markers
%
for i3 = 1:length(vars.Opals)
    %
    % tumor antibody should have been labeled with 'Tumor' in inForm
    %
    a(:,i3+1) = cellfun(@(x,y)strcat(...
        x,Markers.all{i3},'_Opal',num2str(vars.Opals(i3)),'_',y,...
        '_NormalizedCounts_TotalWeighting_'),vars.comp,vars.type,'Uni',0);
    %
    % However, mistakes happen so we check with the Tumor
    % antibody designation from inForm
    %
    a2(:,i3+1) = cellfun(@(x,y)strcat(...
        x,Markers.all_original{i3},'_Opal',num2str(vars.Opals(i3)),'_',y,...
        '_NormalizedCounts_TotalWeighting_'),vars.comp,vars.type,'Uni',0);
    %
    % Do the same for alternate DAPI naming
    %
    a3(:,i3+1) = a(:,i3+1);
    a4(:,i3+1) = a2(:,i3+1);
    %
    b(:,i3+1) = cellfun(@(x,y)strcat(y,x,num2str(vars.Opals(i3))),...
        vars.comp,vars.type,'Uni',0);
end
%
% put the strings into a single vector
%
zz = a';
zz = vertcat(zz(:))';
vars.namesin = cellfun(@(x)shortenName(x),zz,'Uni',0);
%
zz = a2';
zz = vertcat(zz(:))';
vars.namesin_original = cellfun(@(x)shortenName(x),zz,'Uni',0);
%
zz = a3';
zz = vertcat(zz(:))';
vars.namesin_dapierror = cellfun(@(x)shortenName(x),zz,'Uni',0);
%
zz = a4';
zz = vertcat(zz(:))';
vars.namesin_dapierror_original = cellfun(@(x)shortenName(x),zz,'Uni',0);
%
% New Table Names
%
zz = b';
zz = vertcat(zz(:))';
vars.names = zz;
%
% read the text files in each folder to a cell vector (read lineage and
% expr separate?)
%
[v,units] = cellfun(@(x) readtxt(...
    filename, x, vars, wd, layers), Markers.lin, 'Uni', 0);
[v2,units2] = cellfun(@(x) readtxt(...
    filename, x, vars, wd, layers), Markers.expr, 'Uni', 0);
%
[~, loc] = ismember(Markers.lin, Markers.all);
unit_name = repmat({'pixels'}, length(Markers.all),1);
unit_name(loc) = units;
%
[~, loc] = ismember(Markers.expr, Markers.all);
unit_name(loc) = units2;
units = unit_name;
%
% read in tables for multiple segmentation
%
idx = find(Markers.nsegs > 1);
idx_count = 0;
v3 = {};
%
if idx
   for i1 = 1:length(idx)
        cidx = idx(i1);
        for i2 = 2:Markers.nsegs(cidx)
            idx_count = idx_count + 1;
            x = [Markers.all{cidx},'_',num2str(i2)];
            [v3{idx_count},units3] = readtxt(filename,x, vars, wd, layers); %#ok<AGROW>
            if ~strcmp(units3, units(cidx))
                e_code = 16;
                return
            end
        end
   end
end
%
% merge cell vectors
%
v = [v,v2,v3];
%
% convert names to something more understandable
%
for i4 = 1:length(v)
    v{i4}.Properties.VariableNames = ...
        [{'SampleName','SlideID','fx','fy',...
        'CellID','Phenotype','CellXPos',...
        'CellYPos','EntireCellArea'},vars.names{:}];
end
%
% get the lineage tables out of v and put them into struct C
%
for i3 = 1:length(Markers.lin)
    C.(Markers.lin{i3}) = v{i3};
end
%
% get expr markers out of v and put them into struct C, merging tables for 
% antibodies with multiple segmentations
%
ii = ismember(Markers.all,Markers.expr);
nsegs = Markers.nsegs(ii);
i4 = 1;
i6 = 0;
%
for i3 = length(Markers.lin)+1:length(Markers.all)
    if nsegs(i4) > 1
        tbl = v{i3};
        for i5 = 2:nsegs(i4)
            i6 = i6 + 1;
            v3_count = length(Markers.all) + i6;
            tbl = [tbl;v{v3_count}]; %#ok<AGROW>
        end
        C.(Markers.expr{i4}) = tbl;
    else
        C.(Markers.expr{i4}) = v{i3};
    end
    i4 = i4+1;
end
%
% add filenames to data struct
%
C.fname = filename;
%
end
%% function: readtxt;
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description:
% reads in a single cell seg data file
%% --------------------------------------------------------
%%
function [Q, units] = readtxt(filnm,marker,vars,wd,layers)
%%-----------------------------------------------------------
%% load the csv file with the given marker
%%-----------------------------------------------------------
%
% read in table
%
warning('off','MATLAB:table:ModifiedAndSavedVarnames')
%
% create format specifier for each column in the table
%
formatspec = strcat(repmat('%s ',[1,4]),{' '},repmat('%f32 ',[1,11]),...
    { ' %s '},repmat('%f32 ',[1,5]),{' '},repmat('%f32 ',[1,5*layers]),...
    { ' %s '},repmat('%f32 ',[1,5]),{' '},repmat('%f32 ',[1,5*layers]),...
    { ' %s '},repmat('%f32 ',[1,5]),{' '},repmat('%f32 ',[1,5*layers]),...
     { ' %s '},repmat('%f32 ',[1,4]),{' '},repmat('%f32 ',[1,5*layers]),...
    {' '},repmat('%s ',[1,2]),{' '}, repmat('%f32 ',[1,4]),{' '}, ....
    repmat('%s ',[1,2]));
formatspec = formatspec{1};
%
try
    T = readtable([wd,'\Phenotyped\',marker,'\',filnm.name],'Format',formatspec,...
        'Delimiter','\t','TreatAsEmpty',{' ','#N/A'});
catch
    nm = extractBefore(filnm.name,"]_cell_seg");
    if isempty(nm)
        nm = extractBefore(filnm.name,"]_CELL_SEG");
    end
    try
        nm1 = [nm, ']_CELL_SEG_DATA.TXT'];
        T = readtable([wd,'\Phenotyped\',marker,'\',nm1],'Format',formatspec,...
            'Delimiter','\t','TreatAsEmpty',{' ','#N/A'});
    catch
         nm1 = [nm, ']_cell_seg_data.txt'];
         T = readtable([wd,'\Phenotyped\',marker,'\',nm1],'Format',formatspec,...
        'Delimiter','\t','TreatAsEmpty',{' ','#N/A'});
    end
end
%
% Find Coordinates
%
f = strsplit(filnm.name,{'_[',']_'});
f = strsplit(f{2},',');
T.fx = repmat(str2double(f{1}),height(T),1);
T.fy = repmat(str2double(f{2}),height(T),1);
colnames = T.Properties.VariableNames;
%
if any(contains(colnames,'pixels'))
    units = 'pixels';
    areavariable = 'EntireCellArea_pixels_';
else
    units = 'microns';
    areavariable = 'EntireCellArea_squareMicrons_';
end
%
% Selects variables of interest from originial data table by name
%
namestry(1,:) = [{'SampleName','SlideID','fx','fy',...
    'CellID','Phenotype','CellXPosition',...
    'CellYPosition'},areavariable,...
    vars.namesin{:}];
namestry(2,:) = [{'SampleName','SlideID','fx','fy',...
    'CellID','Phenotype','CellXPosition',...
    'CellYPosition'},areavariable,...
    vars.namesin_original{:}];
namestry(3,:) =  [{'SampleName','SlideID','fx','fy',...
    'CellID','Phenotype','CellXPosition',...
    'CellYPosition'},areavariable,...
    vars.namesin_dapierror{:}];
namestry(4,:) =  [{'SampleName','SlideID','fx','fy',...
    'CellID','Phenotype','CellXPosition',...
    'CellYPosition'},areavariable,...
    vars.namesin_dapierror_original{:}];
%
fail = 1;
icount = 1;
while fail == 1
    try
        Q = T(:,namestry(icount,:));
        fail = 0;
    catch
        fail = 1;
    end
    %
    icount = icount + 1;
    if icount > length(namestry(:,1))
        fail = 0;
    end
end
%
end
%% function:shortenName; shorten the names of the inForm columns
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% If the names of the columns in inForm are longer than 63 char then
% matlab will only read the first 63 char of the header and the search
% strings need to be adjusted as such. This function takes in a string,
% checks if it is longer than 63 chars and, if it is, cuts the string at
% 63 chars
%% --------------------------------------------------------------
%%
function out = shortenName(n)
out = n;
if length(out) > 63
    out = extractBefore(out,64);
end
end
%% function: getphenofield; 
%% --------------------------------------------------------------
%% Created by: Alex Szalay
% Last Edit: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% separate the different phenotype for each ABxx stuctures so that they
% only contain the positive phenotypes; get .tmp to be used for the Other
% classification based on the lowest Opal in the first segmentation
% algorithm; add the file name as an extra field in the output structure
%% --------------------------------------------------------
%%
function f = getphenofield(C, Markers, units)
%
% get the X and Y resolution + image size for unit conversions 
%
fold = extractBefore(C.fname.folder,'Phenotyped');
fold = [fold,'\Component_Tiffs'];
iname = [fold,'\',extractBefore(C.fname.name,...
    "]_cell_seg"),']_component_data.tif'];
if isempty(iname)
    iname = [fold,'\',extractBefore(C.fname.name,...
        "]_CELL_SEG"),']_component_data.tif'];
end
%
imageinfo = imfinfo(iname);
W = imageinfo.Width;
H = imageinfo.Height;
scalea = 10^4 *(1/imageinfo(1).XResolution); % UM/PX
%
% get only the postive cells from each phenotype
%
for i3 = 1:length(Markers.all)
    %
    mark = Markers.all{i3};
    mark1 = lower(mark);
    %
    Markers.lall{i3} = mark1;
    %
    dat = C.(mark);
    dat2 = dat(strcmpi(dat.Phenotype,mark),:);
    %
    if strcmp(mark,'Tumor') && isempty(dat2)
        markb = Markers.all_original{i3};
        dat2 = dat(strcmpi(dat.Phenotype,markb),:);
    end
    %
    if ~isempty(dat2)
        %
        dat2.Phenotype = repmat({mark},height(dat2),1);
        %
        if strcmp(units{i3},'microns')
            fx = (dat2.fx - scalea*(W/2)); %microns
            fy = (dat2.fy - scalea*(H/2)); %microns
            dat2.CellXPos = floor(1/scalea .* (dat2.CellXPos - fx));
            dat2.CellYPos = floor(1/scalea .* (dat2.CellYPos - fy));
        end
        %
    end
    f.(mark1) = dat2;
end
%
% create a set of others
%
dat2 = C.(Markers.seg{1});
[~,loc] = ismember(Markers.seg, Markers.all);
%
if ~isempty(dat2)
    %
    dat2.Phenotype = repmat({'Other'},height(dat2),1);
    %
    if strcmp(units{loc}, 'microns')
        fx = (dat2.fx - scalea*(W/2)); %microns
        fy = (dat2.fy - scalea*(H/2)); %microns
        dat2.CellXPos = floor(1/scalea .* (dat2.CellXPos - fx));
        dat2.CellYPos = floor(1/scalea .* (dat2.CellYPos - fy));
    end
    %
end
f.tmp = dat2;
%
f.fname = C.fname;
end
%% function: getdistinct;
%% --------------------------------------------------------------
%% Created by: Alex Szalay
%%% Last Edit: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% define the Other cells;start by getting all cells from the lowest 
% expression marker opal in the 1ry segmentation
% eliminate all the cells within 6 pixels of a lineage classification,
% for 'Other' cells that are within 6 pixels of each other remove cells
% based on
% 1. whether the cell is to close to two cells 
% 2. based on second expression marker intensity
%% --------------------------------------------------------
%%
function q = getdistinct(d,Markers)
%
% check for cells within 6 pixels of 'Other' cells
%
w = 6;
%
d.tmp = [d.tmp; d.(lower(Markers.seg{1}))];
e = cellfun(@(x)min(edist(d.(lower(x)),d.tmp),[],1)',Markers.lin, 'Uni', 0);
%
for i1 = 1:length(Markers.lin)
    if isempty(e{i1})
        e{i1} = repmat(w+1,[height(d.tmp) 1]);
    end
end
%
e = [e{:}];
ii = sum(e <= w, 2);
ii = ii ~= 0;
d.tmp = d.tmp(~ii,:);
%
%
% remove the Others within 6 pixels of each other
%
r = edist(d.tmp,d.tmp);
r0 = (r<=6.0);
n0 = size(r0,1);
%
r0 = r0.*(1-eye(n0));
r0 = triu(r0);
[ix,iy] = find(r0>0);
%
% delete cells that are in the table more than once (ie a cell that is too
% close to two cells)
%
[~,b] = unique(iy);
rows = ismember(iy,iy(setdiff(1:length(iy),b)));
qx = unique(iy(rows));
d.tmp.Phenotype(qx,:) = {'Coll'};
iy = iy(~rows);
ix = ix(~rows);
%
% delete the rest of the cells based on the sum of expression marker profile
%
expr = Markers.Opals(ismember(Markers.all,Markers.expr));
ixe =  zeros(length(ix), length(Markers.expr));
iye =  zeros(length(iy), length(Markers.expr));
%
for i1 = 1:length(Markers.expr)
    opnm = num2str(expr(i1));
    tnm = ['MeanEntireCell',opnm];
    ixe(:,i1) = table2array(d.tmp(ix,tnm));
    iye(:,i1) =  table2array(d.tmp(iy,tnm));
end
%
rows =  sum(ixe,2) >= sum(iye,2);
%
qx = iy(rows);
d.tmp.Phenotype(qx,:) = {'Coll'};
%
qx = ix(~rows);
d.tmp.Phenotype(qx,:) = {'Coll'};
%
qx = ~strcmp(d.tmp.Phenotype,'Coll');
d.tmp = d.tmp(qx,:);
%
d.tmp.Phenotype = repmat(cellstr('Other'),height(d.tmp),1);
%
% make the next struct for coex%%%
%
q.other = d.tmp;
q.all = [];
for i3 = 1:length(Markers.lin)
    q.all = [q.all;d.(lower(Markers.lin{i3}))];
end
%
% keep original phenotype
%
q.og = [q.all; q.other];
%
q.fname = d.fname;
for i3 = 1:length(Markers.expr)
    mark = lower(Markers.expr{i3});
    q.(mark) = d.(mark);
end

end
%% function: getcoex; 
%% --------------------------------------------------------------
%% Created by: Alex Szalay
% Last Edit: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% resolve the cells with multiple lineage marker calls within 6 pixels of
% each other. This is done in three steps:
% 1. remove cells that are of the same lineage within 6 pixels of each 
% other based off of expression marker intensity;
% 2. second change accepted coexpression to [1st Opal Target name, 2nd 
% Opal Target name]; 
% 3. remove cells based off of the segmentation hierarchy in the table 
%% --------------------------------------------------------------
%%
function q = getcoex(d,Markers)
%
% compute distance matrix with overlapping cells
%
[c] = createoverlapmatrix(d.all);
%
q = d;
q.c = c;
%
% remove missegmented cells
%
[q,c] = remvover(q,c,Markers);
%
% find coexpression after removal of missegmented cells & replace those
% calls in segmentation table 
%
[q,~] = addcoex(q,c,Markers);
%
% recompute the distance matrix
%
[c2] = createoverlapmatrix(q.all);
%
% patch for multiple coexpressions
%
[q,~,Markers] = addmultiplecoex(q,c2,Markers);
%
% recompute the distance matrix
%
[c2] = createoverlapmatrix(q.all);
%
% remove double calls coex
%
[q,~] = remvcoex(q,c2,Markers);
%
q.fig = [q.all;q.other];
qx = ~strcmp(q.fig.Phenotype,'Coll');
q.fig = q.fig(qx,:);
%
q = rmfield(q,'other');
%
% check again there are any cells too close in the pixel boundary
%
[c2] = createoverlapmatrix(q.fig);
%
if ~isempty(c2)
    error('Cells within limit, 6 pixels, exist after cleanup')
end
%
end
%% function: createoverlapmatrix;
%% --------------------------------------------------------------
%% Created by: Alex Szalay - Johns Hopkins 
%% Editied by: Benjamin Green Johns Hopkins - 11/15/2019
%% --------------------------------------------------------------
%% Description
% function to compute the cells within 6 pixels of each other
%% --------------------------------------------------------------
%%
function [c] = createoverlapmatrix(d)
%
% compute distance matrix
%
e  = edist(d,d);
%
% filter on e<=6pixels
%
e0 = (e<=6.0);
n02 = size(e0,1);
%
% remove diagonal, and get upper triangle
%
e0 = e0.*(1-eye(n02));
e0 = triu(e0);
[ix2,iy2] = find(e0>0);
%
% assemble it to coex
%
x = d.CellXPos;
y = d.CellYPos;
p = d.Phenotype;
%
c = table(x(ix2),y(ix2),p(ix2),x(iy2),y(iy2),p(iy2),ix2,iy2);
c.Properties.VariableNames = {'x1','y1','p1','x2','y2','p2','i1','i2'};
c.dist = sqrt(double((c.x1-c.x2).^2+(c.y1-c.y2).^2));
%
end
%% function: remvover;
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% remove cells of the same type which are too close together based off
% of:
% 1. whether the cell is a part of an acceptable coexpression
% 2. based off of total expression marker intensity 
%% -------------------------------------------------------------
%%
function[q,c] = remvover(q,c,Markers)
%
% find cells of same type which are too close together
%
rows = strcmp(c.p1,c.p2);
q.flags.misseg = c(rows,:);
q.flags.misseg.c = find(rows>0);
%
for i2 = 1:length(Markers.add)
ii = Markers.add(i2);
%
SW = cellfun(@(x)startsWith(ii,x), Markers.all,'Uni',0);
SW = [SW{:}];
SWm = Markers.all{SW};
%
EW = cellfun(@(x)endsWith(ii,x), Markers.all,'Uni',0);
EW = [EW{:}];
EWm = Markers.all{EW};
%
% find coexpression before removal of same type cells
%
rows = (strcmp(c.p1,SWm) & strcmp(c.p2,EWm))|...
    (strcmp(c.p1,EWm) & strcmp(c.p2,SWm));
q.coex = c(rows,:);
q.coex.c = find(rows>0);
%
midx1 = q.flags.misseg.i1;
cidx1 = q.coex.i1;
midx2 = q.flags.misseg.i2;
cidx2 = q.coex.i2;
%
rows1 = ismember(midx1, cidx1)|ismember(midx1,cidx2);
rows2 = ismember(midx2, cidx1)|ismember(midx2,cidx2);
%
% if cells are too close together keep the cell if it is coepressed with
% another cell and delete the one that is not coexpressed
%
rows = rows1==1 & rows2==0;
%
q.all.Phenotype(q.flags.misseg.i2(rows))= {'Coll'};
c.p2(q.flags.misseg.c(rows)) = {'Coll'};
q.flags.misseg.p2(rows) = {'Coll'};
%
rows =  rows1 == 0 & rows2 == 1;
q.all.Phenotype(q.flags.misseg.i1(rows))= {'Coll'};
c.p1(q.flags.misseg.c(rows)) = {'Coll'};
q.flags.misseg.p1(rows) = {'Coll'};
%

end

[~,ii] = ismember(Markers.expr,Markers.all);
exprO = Markers.Opals(ii);
%
for i1 = 1:length(Markers.all)
    mark = zeros(length(Markers.all),1);
    mark(i1) = 1;
    ii = zeros(length(Markers.all), length(Markers.expr));
    for i2 = 1:length(Markers.Coex)
        ii(:,i2) = mark & Markers.Coex{i2};
    end
    ii = sum(ii,1);
    ii = logical(ii);
    %
    if sum(ii)
        %
        expr = arrayfun(@(x)['MeanMembrane',num2str(x)],exprO,'Uni',0);
        %
        mark = Markers.all{i1};
        rows = find(strcmp(q.flags.misseg.p1,mark));
        %
        t = table2array(q.all(:,q.all.Properties.VariableNames(expr)));
        %
        % get i1 expression total
        %
        midx1 = q.flags.misseg.i1(rows);
        eidx1 = t(midx1,ii);
        eidx1 = sum(eidx1,2);
        %
        % get i2 expression total
        %
        midx2 = q.flags.misseg.i2(rows);
        eidx2 = t(midx2,ii);
        eidx2 = sum(eidx2,2);
        %
        % get rows where i1 expression is more than i2 expression
        %
        rows4 = eidx1 > eidx2;
        rows1 = rows(rows4);
        %
        q.all.Phenotype(q.flags.misseg.i2(rows1))= {'Coll'};
        c.p2(q.flags.misseg.c(rows1)) = {'Coll'};
        q.flags.misseg.p2(rows1) = {'Coll'};
        %
        %
        % check that missegmented cell is not duplicated/ causing other
        % invalid coexpression
        %
        ii1 = q.flags.misseg.c(rows1);
        r = ismember(c.i2,c.i2(ii1));
        c.p2(r) = {'Coll'};
        r = ismember(c.i1,c.i2(ii1));
        c.p1(r) = {'Coll'};
        %
        rows1 = rows(~rows4);
        %
        q.all.Phenotype(q.flags.misseg.i1(rows1))= {'Coll'};
        c.p1(q.flags.misseg.c(rows1)) = {'Coll'};
        q.flags.misseg.p1(rows1) = {'Coll'};
        %
        % check that missegmented cell is not duplicated/ causing other
        % invalid coexpression
        %
        ii1 = q.flags.misseg.c(rows1);
        r = ismember(c.i2,c.i1(ii1));
        c.p2(r) = {'Coll'};
        r = ismember(c.i1,c.i1(ii1));
        c.p1(r) = {'Coll'};
    end
end
end
%% function: addcoex; 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% find the acceptable coexpression after cells that are too close
% together are resolved; rename the cells to what the acceptable
% coexpression name is in the table; remove the cell which the acceptable
% coexpression marker designation in Tables ends with and keep the cell
% which the acceptable coexpression marker designation in Tables starts
% with
%% --------------------------------------------------------------
%%
function [q,c] = addcoex(q,c,Markers)
q.coex = table();
for i2 = 1:length(Markers.add)
    ii = Markers.add(i2);
    %
    SW = cellfun(@(x)startsWith(ii,x), Markers.all,'Uni',0);
    SW = [SW{:}];
    SWm = Markers.all{SW};
    %
    EW = cellfun(@(x)endsWith(ii,x), Markers.all,'Uni',0);
    EW = [EW{:}];
    EWm = Markers.all{EW};
    %
    % find coexpression and replace whatever the additional call starts 
    % with(ie. lower Opal); remove whatever the additional call ends with
    % (ie. higher Opal)
    %
    rows = strcmp(c.p1,SWm) & strcmp(c.p2,EWm);
    if sum(rows) > 0
        q.all.Phenotype(c.i1(rows))= {'Coll'};
        q.all.Phenotype(c.i2(rows))= ii;
        q.coex = [q.coex;c(rows,:)];
        c.p1(rows) = {'Coll'};
        c.p2(rows) = ii;
    end
    %
    rows = strcmp(c.p1,EWm) & strcmp(c.p2,SWm);
    if sum(rows) > 0
        q.all.Phenotype(c.i2(rows))= {'Coll'};
        q.all.Phenotype(c.i1(rows))= ii;
        q.coex = [q.coex;c(rows,:)];
        c.p2(rows) = {'Coll'};
        c.p1(rows) = ii;
    end
    %
end
end
%% function: remvcoex; 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% remove the cells within 6 pixels of each which have conflicting lineage
% marker designations (lineage markers not in acceptable coexpression)
% based off of the segmentation hierarchy
%% --------------------------------------------------------------
%%
function [q,c2] = remvcoex(q,c2,Markers)
%
seg = cellfun(@str2double,Markers.SegHie);
seg1 = seg;
%
allmarkers = [Markers.lin, Markers.add];
while ~isempty(seg1)
    %
    s1 = max(seg1);
    seg1(seg1 == s1) = [];
    %
    ii = seg == s1;
    % 
    mark = allmarkers(ii);
    %
    for i1 = 1:length(mark)
        marka = mark{i1};
        rows = strcmp(c2.p2,marka) & ~strcmp(c2.p1,'Coll');
        q.all.Phenotype(c2.i2(rows))= {'Coll'};
        q.flags.(marka) = c2(rows,:);
        c2.p2(rows) = {'Coll'};
        %
        rows = strcmp(c2.p1,marka) & ~strcmp(c2.p2,'Coll');
        q.all.Phenotype(c2.i1(rows)) = {'Coll'};
        q.flags.(marka) = c2(rows,:);
        c2.p1(rows) = {'Coll'};
    end
    %
end
end
%% function: addmultiplecoex;
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 11/15/2019
%% --------------------------------------------------------------
%% Description
% for coexpression more than 2 cells 
%% --------------------------------------------------------------
%%
function [q,c2,Markers] = addmultiplecoex(q,c2,Markers)
%
seg = cellfun(@str2double,Markers.SegHie);
seg1 = seg;
%
allmarkers = [Markers.lin, Markers.add];
while ~isempty(seg1)
    %
    s1 = min(seg1);
    seg1(seg1 == s1) = [];
    %
    ii = seg == s1;
    % 
    mark = allmarkers(ii);
    if numel(mark) > 4
        ii = ~ismember(mark,Markers.add);
        markb = flip(mark(ii));
        markb = strcat(markb{:});
        mark = mark(~ii);
        %
        %
        for i1 = 1:length(mark)
            marka = mark{i1};
            rows = strcmp(c2.p2,marka) & ismember(c2.p1,mark);
            q.all.Phenotype(c2.i1(rows))= {'Coll'};
            q.all.Phenotype(c2.i2(rows))= {markb};
            q.flags.(marka) = c2(rows,:);
            c2.p1(rows) = {'Coll'};
            c2.p2(rows) = {markb};
            %
            rows = strcmp(c2.p1,marka) & ismember(c2.p2,mark);
            q.all.Phenotype(c2.i2(rows)) = {'Coll'};
            q.all.Phenotype(c2.i1(rows))= {markb};
            q.flags.(marka) = c2(rows,:);
            c2.p2(rows) = {'Coll'};
            c2.p1(rows) = {markb};
        end
        Markers.add = [Markers.add,markb];
        Markers.SegHie = [Markers.SegHie,num2str(s1)];
        %
    end
    %
end
end
%% function: getseg;
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% remove other cells with cell centers within non primary segmentation 
% boundaries; If a cell is on the boundary of the tissue the algorithm
% will close the cell to complete the calculation
%% --------------------------------------------------------------
%%
function [p] = getseg(q,Markers)
p = q;
%
% find the rows which do not have a membrane
%
mrows = isnan(p.fig.MeanMembraneDAPI);
nmr = p.fig(mrows,:);
p.fig = p.fig(~mrows,:);
%
% make global cellids for the image before segmentation correction
%
CellID = (1:1:height(p.fig))';
p.fig.Properties.VariableNames('CellID') = {'CellNum'};
p.fig = [table(CellID),p.fig];
%
% find the other cells before the tumor segmentation correction
%
orows = find(strcmp(p.fig.Phenotype, 'Other'));
%
% get segmentation outlines for all alternative segmentations
%
im3 = cell(length(Markers.altseg));
tcellids = cell(length(Markers.altseg));
trows = false(max(CellID),length(Markers.altseg));
%
for i1 = 1:length(Markers.altseg)
    %
    markalt = Markers.altseg{i1};
    idx = ismember(Markers.all, markalt);
    %
    SS = Markers.SegStatus(idx);
    s_markers_idx = Markers.SegStatus == SS & ismember(Markers.all,...
        Markers.lin)';
    s_markers = Markers.all(s_markers_idx);
    if ~isempty(Markers.add)
        iis = startsWith(Markers.add, s_markers);
        s_markers = [s_markers, Markers.add(iis)];
    end
    %
    % get folder and image names for altseg
    %
    fdname = [extractBefore(p.fname.folder,Markers.all{1}),...
        markalt,'\'];
    iname = [fdname,extractBefore(p.fname.name,...
        "]_cell_seg"),']_binary_seg_maps.tif'];
    if isempty(iname)
        iname = [fdname,extractBefore(p.fname.name,...
            "]_CELL_SEG"),']_binary_seg_maps.tif'];
    end
    %
    % get rows of altseg cells
    %
    trows(:,i1) = ismember(p.fig.Phenotype, s_markers);
    %
    % get inForm cellids of altseg cells
    %
    tcellids{i1} = double(p.fig.CellNum(trows(:,i1),:));
    %
    % read in the image for segmentation 
    %
    im = imread(iname,4);
    %
    % convert it to a linear index of labels
    %
    im = label2idx(im);
    im3{i1} = im(1,tcellids{i1});
end
%
%get filenames for 1ry seg images
%
iname = fullfile(p.fname.folder,p.fname.name);
iname = replace(iname, Markers.all{1}, Markers.seg{1});
iname = [extractBefore(iname,"]_cell_seg"),']_binary_seg_maps.tif'];
if isempty(iname)
    iname = [extractBefore(iname,"]_CELL_SEG"),']_binary_seg_maps.tif'];
end
%
% get cellids of 1ry seg cells
%
trowsall = sum(trows,2) > 0;
cellids = double(p.fig.CellNum(~trowsall,:));
%
% read in image convert to linear index 
%
im2 = imread(iname,4);
im4 = label2idx(im2);
%
% get image dimensions
%
s = {size(im2)};
p.size.T = 2; p.size.B = s{1}(1)-1; p.size.L = 2; p.size.R = s{1}(2)-1;
%
% Remove non 1ry cells
%
im4 = im4(1,cellids);
%
% get expected size of cell vector
%
s2 = max(CellID);
%
% compile the segmentation images
%
im5 = cell(1,s2);
%
% first input all altsegs
%
for i1 = 1:length(Markers.altseg)
    im5(CellID(trows(:,i1))) = im3{i1};
end
%
% next input the 1ry segmentations
%
im5(CellID(~trowsall)) = im4;
%
% fill the cells and get them in the proper format
%
[obj,s] = OrganizeCells(p,s,s2,im5);
%
% remove others in altsegs
%
objt = obj(trowsall);
objt = cat(1,objt{:});
%
X = p.fig.CellXPos(orows) + 1;
Y = p.fig.CellYPos(orows) + 1;
Oth = sub2ind(s{1},Y,X);
rows = ismember(Oth,objt);
%
% count number of deleted others
%
p.flags.segclean = sum(rows == 1);
%
% remove those others from the output p.fig
%
p.fig(orows(rows),:) = [];
%
% remove those others from the CellID vector
%
CellID(orows(rows)) = [];
%
% remove those others from the cell objects variable and save to p.obj
%
p.obj = obj(CellID);
%
% add on those cells without membranes to the bottom of p.fig
%
nmr.CellNum = nmr.CellID; 
p.fig = vertcat(p.fig,nmr);
%
% get new and final CellIDs
%
CellID = (1:1:height(p.fig))';
p.fig.CellID = CellID;
%
end
%% function: OrganizeCells;
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% close the cells on the edge of the tissue and organize the numbers
% inside the object cell to a circle so that the inside of the cell can
% be filled
%% --------------------------------------------------------------
%%
function [obj,s] = OrganizeCells(p,s,s2,im5)
%
% convert linear index to x,y
%
s = repmat(s,1,s2);
[y,x] = cellfun(@ind2sub,s,im5,'Uni',0);
%
% locate cells on edge of tissue & fill edges of cells on edge of image
%
X = cellfun(@min,x);
Y = cellfun(@min,y);
Y2 = cellfun(@max,y);
X2 = cellfun(@max,x);
rows = find(Y == p.size.T|Y2 == p.size.B|X == p.size.L|X2==p.size.R);
[x(rows),y(rows)] = cellfun(@(x,y)...
    filledges([x,y],p.size.T,p.size.B,p.size.L,p.size.R),...
    x(1,rows),y(1,rows),'Uni',0);
%
% find midpoints of the cell objects
%
mx = cellfun(@mean,x);
my = cellfun(@mean,y);
%
% get the angle of each point from the midpoint
%
ang = cellfun(@(x,y,mx,my)atan2(y-my,x-mx),...
    x,y,num2cell(mx),num2cell(my),'Uni',0);
%
% sorting the angles from the midpoint we can get indicies for a circle
%
[~,o] = cellfun(@sort,ang,'Uni',0);
%
% order the cells in the circle
%
x = cellfun(@(x,o)x(o),x,o,'Uni',0);
y = cellfun(@(y,o)y(o),y,o,'Uni',0);
%
% put the cells in reference coordinate rectangles
%
x1 = cellfun(@(x,y)x-y,x,num2cell(X),'Uni',0);
y1 = cellfun(@(x,y)x-y,y,num2cell(Y),'Uni',0);
%
% fill objects
%
obj = arrayfun(@(X,X2,Y,Y2)zeros(Y2-Y+1,X2-X+1),X,X2,Y,Y2,'Uni',0);
obj = cellfun(@fillobj,obj,y1,x1,'Uni',0);
[objy,objx] = cellfun(@(x)find(x==1),obj,'Uni',0);
%
% convert back to orginal image coordinates
%
objy = cellfun(@(x,y)x+y-1,objy,num2cell(Y),'Uni',0);
objx = cellfun(@(x,y)x+y-1,objx,num2cell(X),'Uni',0);
obj = cellfun(@sub2ind,s,objy,objx,'Uni',0);
%
% if the cell is very small then the obj is written as a column vector
% instead of a row vector
% to fix this find those cells
%
[~,r2] = cellfun(@size,obj,'Uni',0);
r2 = find(cat(1,r2{:})~=1);
%
% check if any of these cells exist
%
if ~isempty(r2)
    %
    % correct them to proper dimensions
    %
    for i4 = 1:length(r2)
        obj{r2(i4)} =  obj{r2(i4)}';
    end
end
end
%% function: filledges; 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% this function takes in a cell object bon the edge of the image and 
% fills in the boundaries using T,B,L,R to define the edges as Top,
% Bottom, Left, Right
%% --------------------------------------------------------------
%%
function[x,y] = filledges(b,T,B,L,R)
w = 1;
tedge = [];
%
% get a vector that defines if the cell is on the top or bottom of the
% image
%
rows = b(:,2) == T |  b(:,2) == B;
edges = b(rows,:);
%
% find the edge where its pixel nieghborhood is missing a boundary
%
for i1 = 1:length(edges(:,1))
    x = edges(i1,:);
    neighbor=zeros(8,2);
    neighbor(1,:) = [x(1)-1, x(2)-1];
    neighbor(2,:) = [x(1),x(2)-1];
    neighbor(3,:) = [x(1)+1,x(2)-1];
    neighbor(4,:) = [x(1)-1,x(2)];
    neighbor(5,:) = [x(1)+1,x(2)];
    neighbor(6,:) = [x(1)-1,x(2)+1];
    neighbor(7,:) = [x(1),x(2)+1];
    neighbor(8,:) = [x(1)+1,x(2)+1];
    neighborhood(1) = sum(ismember(...
        neighbor([1,4,6],:),b,'rows'))>0;
    neighborhood(2) = sum(ismember(...
        neighbor([7,2],:),b,'rows'))>0;
    neighborhood(3) = sum(ismember(...
        neighbor([3,5,8],:),b,'rows'))>0;
    val = sum(neighborhood);
    if val == 1
        tedge(w,:) = x;
        w = w+1;
    end
end
%
% get a vector that defines if the cell is on the top or bottom of the
% image
%
rows = b(:,1) == L| b(:,1) ==  R;
edges = b(rows,:);
%
% find the edge where its pixel nieghborhood is missing a boundary
%
for i1 = 1:length(edges(:,1))
    x = edges(i1,:);
    neighbor=zeros(8,2);
    neighbor(1,:) = [x(1)-1, x(2)-1];
    neighbor(2,:) = [x(1),x(2)-1];
    neighbor(3,:) = [x(1)+1,x(2)-1];
    neighbor(4,:) = [x(1)-1,x(2)];
    neighbor(5,:) = [x(1)+1,x(2)];
    neighbor(6,:) = [x(1)-1,x(2)+1];
    neighbor(7,:) = [x(1),x(2)+1];
    neighbor(8,:) = [x(1)+1,x(2)+1];
    neighborhood(1) = sum(ismember(...
        neighbor([6,7,8],:),b,'rows'))>0;
    neighborhood(2) = sum(ismember(...
        neighbor([4,5],:),b,'rows'))>0;
    neighborhood(3) = sum(ismember(...
        neighbor([1,2,3],:),b,'rows'))>0;
    val = sum(neighborhood);
    if val == 1
        tedge(w,:) = x;
        w = w+1;
    end
end
%
% close the cell if it touches the edge
%
ns1 = [];ns2=[];
if ~isempty(tedge)&& length(tedge(:,1)) == 2 
    rows = tedge(:,1) == T | tedge(:,1) == B;
    %
    m1 = min(tedge(:,2))+1;
    m2 = max(tedge(:,2))-1;
    ns1(:,2) = m1:m2;
    %
    m1 = min(tedge(:,1))+1;
    m2 = max(tedge(:,1))-1;
    ns2(:,1) = m1:m2;
    %
    if sum(rows) == 0
        ns2(:,2) = tedge(1,2);
        ns = ns2;
    elseif sum(rows) == 2
        ns1(:,1) = tedge(1,1);
        ns = ns1;
    else
        ns1(:,1) = tedge(rows,1);
        ns2(:,2) = tedge(~rows,2);
        ns = vertcat(ns1,ns2);
    end
    %
    b = vertcat(b,ns);
    %
end
x = b(:,1);
y = b(:,2);
end
%% function: fillobj; 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% fill the cell object 
%% --------------------------------------------------------------
%%
function[obj] = fillobj(obj,y,x)
s = size(obj);
x = x+1;
y = y+1;
obj(sub2ind(s,y,x)) = 1;
obj = imfill(obj);
end
%% function: getexprmark; 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% determine the expression marker status of each cell; if the cell is in
% two cell membranes choose the cell that the expression marker call is
% closest to 
%% --------------------------------------------------------------
%%
function o = getexprmark(q,Markers)
o = q;
if ~isempty(o.fig)
    ii = ismember([480;520;540;570;620;650;690;780],Markers.Opals);
    bins = [2,4,8,16,32,64,128,256];
    bins(~ii) = [];
    ii = ismember(Markers.all,Markers.expr);
    bins = bins(ii);
    %
    ExprPhenotype = zeros(height(o.fig),1);
    for i3 = 1:length(Markers.expr)
        mark = lower(Markers.expr{i3});
        % 
        % get cell indx
        %
        s = [o.size.B+1,o.size.R+1];
        mx = o.(mark).CellXPos + 1;
        my = o.(mark).CellYPos + 1;
        ii = sub2ind(s, my, mx);
        xx1 = zeros(1,height(o.fig))';
        %
        [x2,x,rows] = findexpr(ii,o);
        %
        % assign cells which are inside two cells or outside of cell 
        % membranes; based on the cell centers distance
        %
        [~,x1] = min(edist(o.fig, o.(mark)(rows,:)),[],1);
        xx1(x1) = 1;
        %
        % for cells located only in one cell apply the expression to that cell
        %
        xx1(x2) = 1;
        %
        ExprPhenotype = ExprPhenotype + (xx1 .* bins(i3));
    end
    o.fig.ExprPhenotype = ExprPhenotype;
else
    o.fig.ExprPhenotype = zeros(0); 
end
%
end
%% function: findexpr; 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% find expression marker calls which are inside two cells or outside of 
% cell membranes based on its cell center location and the mmembrane
% objects
%% --------------------------------------------------------------
%%
function[x2,x3,rows] = findexpr(ii,o)
%
% get the indicies of those rows in expression table that are inside at
% least one cell
%
x = cellfun(@(obj)find(ismember(ii,obj)),o.obj,'Uni',0);
x = cat(1,x{:});
% determine if that cell is in two cells or unique to just one
[a,b] = unique(x);
rows = ismember(x,x(setdiff(1:length(x),b)));

i2 = ii;
i2(a) = [];
i2 = [i2;ii(unique(x(rows)))];
% find those rows in expression table
rows = ismember(ii,i2);
%
% for cells located only in one cell apply the expression to that cell
%
ii = ii(~rows);
x3 = cellfun(@(obj)find(ismember(ii,obj)),o.obj,'Uni',0);
x2 = find(~cellfun(@isempty,x3))';
% if lineage cell has two cells keep the first one for now
x3 = cellfun(@(x)x(1),x3(x2));
end  
%% function: parsave; 
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% write the resulting table using writetable to the *\Table directory
% with the name *_cleaned_Phenotype_table.csv' and if the .csv file has
% more than 60 tumor cells and 600 cells total write the table for the
% ImageLoop for QC output
%% --------------------------------------------------------------
%%
function mktmp = parsave(fData, fname, Markers, wd, imall)
%
% first check that there are enough columns to match the 9 color protocol
%
if width(fData.fig) < 83
    vars.Opals = {'DAPI','480','520','540','570','620','650','690','780'};
    vars.type = [repmat({'Mean'},1,4), repmat({'Total'},1,4)];
    vars.comp = repmat({'Nucleus', 'Membrane','EntireCell','Cytoplasm'},1,2);
    %
    for i3 = 1:length(vars.Opals)
        %
        b(:,i3) = cellfun(@(x,y)strcat(y,x,vars.Opals{i3}),...
            vars.comp,vars.type,'Uni',0);
        %
    end
    %    
    zz = b';
    zz = vertcat(zz(:))';
    vars.names = zz;
    %
    w = fData.fig;
    names_in = w.Properties.VariableNames;
    names_out = [{'CellID','SampleName','SlideID','fx','fy',...
        'CellNum','Phenotype','CellXPos',...
        'CellYPos','EntireCellArea'},vars.names{:},{'ExprPhenotype'}];
    ii = ~ismember(names_out,names_in);
    names_add = names_out(ii);
    vec_add = -1 * ones(height(w), 1); 
    %
    for i4 = 1:length(names_add)
        w.(names_add{i4}) = vec_add;
    end
    w = w(:,names_out);
    %
    % remove NaNs
    %
    for N = vars.names
        temp = num2cell(w.(N{1}));
        temp(cellfun(@isnan,temp)) = {[]};
        w.(N{1}) = temp;
    end
    fData.fig = w;
elseif width(fData.fig) > 83
    disp('Warning: For database upload inForm output should contain ',...
        'no more than 9 color data. Nulls not replaced in tables.');
end
%
% write out whole image csv table
%
nm = [wd,'\Phenotyped\Results\Tables\',extractBefore(fname.name,...
    ']_cell_seg'),']_cleaned_phenotype_table.csv'];
if isempty(nm)
    nm = [wd,'\Phenotyped\Results\Tables\',extractBefore(fname.name,...
    ']_CELL_SEG'),']_cleaned_phenotype_table.csv'];
end
writetable(fData.fig(:,[1,3:end]),nm);
%
% write out image for use in figures later
% saves on images for figures if it is more than 600 cells and 60
% tumor cells in the image (if tumor marker exists)
%
if ~isempty(Markers.Tumor{1})
    r = strcmp(fData.fig.Phenotype,Markers.Tumor{1});
else
    r = ones(height(fData.fig));
end
if (length(fData.fig.CellID) > 400) && (imall || (length(find(r)) > 60))
    fname2 = strcat(wd,'\Phenotyped\Results\tmp_ForFiguresTables\',...
        extractBefore(fname.name,']_cell_seg'),...
        ']_cleaned_phenotype_table.mat');
    if isempty(fname2)
        fname2 = strcat(wd,'\Phenotyped\Results\tmp_ForFiguresTables\',...
            extractBefore(fname.name,']_CELL_SEG'),...
            ']_cleaned_phenotype_table.mat');
    end
    save(fname2,'fData');
    mktmp = 'TRUE';
else
    mktmp = 'FALSE';
end
end
%% function: edist; 
%% --------------------------------------------------------------
%% Created by: Alex Szalay - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% compute the Euclidean distance between the two 2D vectors
%% --------------------------------------------------------------
%%
function e = edist(a,b)
    %
    e = zeros(height(a),height(b));
    for i=1:numel(a.CellXPos)
        X = double(a.CellXPos(i));
        Y = double(a.CellYPos(i));
        e(i,:) = sqrt(double(((b.CellXPos-X).^2+(b.CellYPos-Y).^2)));
    end
end