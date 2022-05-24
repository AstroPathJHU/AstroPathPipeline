%% mkphenim
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% make the phenotype images with all the markers
%% --------------------------------------------------------------
%%
function [q, fullimage, fullimages] = ...
    mkphenim(q, Markers, mycol, imageid, image, simage, doseg)
tp2 = q.fig;
% create composite image
image = image * (mycol.all * 255);
image = uint8(image);
%
imp = reshape(image,[imageid.size,3]);
%
% write out composite image with legend make each AB name the desired
% color in the legend
%
marksa = ['Other',Markers.all];
colsa = 255*mycol.all(1:end-1,:);
%
% to make the text different colors each AB name must be written in a
% separate text box; this loop specifies spacing so each text box starts
% where the last one ends
%
ll = imageid.size(1) - round(imageid.size(1)/ 20);
position = [0, ll];

for i1 = 1:length(marksa)
    cmark = marksa{i1};
    %
    % get last position of the text box
    %
    imp3 = insertText(imp,position(i1,:),cmark,'BoxColor',...
        255*[1,1,1],'BoxOpacity',1,...
        'FontSize',24,'TextColor', 255*[1,1,1]);
    p = imp3(ll,:,:);
    n = p(1,:,:) == 255;
    n = sum(n,3);
    [~,n] = find(n == 3);
    n = max(n);
    %
    % set new position
    %
    position(i1 + 1,:) = [n, ll];
end
position = position(1:end-1,:);
%
% rewrite text box in black and text in correct color
%
imp = insertText(imp,position,marksa,'BoxColor',[0,0,0],...
    'BoxOpacity',1,'FontSize',24,'TextColor', colsa);
%
iname = [imageid.outfull,'composite_image.tif'];
T = Tiff(iname,'w');
T.setTag(imageid.ds);
write(T,imp);
writeDirectory(T)
close(T)
%
fullimage = imp;
fullimages = imp;
%
% add circles for lineage markers
%
radius = 3;
v = .75/radius;
v2 = (radius - 1)/2;
%
marks = [];
cols = [];
%
% add others to lineage calls
%
lincols = [0 0 1 ; mycol.lin];
linmarks = ['Other',Markers.lin];

for i1 = 1: length(linmarks)
    %
    % current marker and color
    %
    curmark = linmarks{i1};
    curcol = lincols(i1,:);
    %
    % x and y positions of cells for current phenotype
    %
    ii = strcmp(tp2.Phenotype,curmark);
    x = tp2(ii,'CellXPos');
    y = tp2(ii, 'CellYPos');
    xy = [x y];
    %
    % create shape array for those phenotypes
    %
    hh = height(xy);
    marks = [marks;table2array(xy), repmat(radius,hh,1)];
    %
    % create color array for those phenotypes
    %
    curcol = uint8(255 * curcol);
    cols = [cols;repmat(curcol,hh,1)];
end
imp = insertShape(imp,'FilledCircle',marks,'Color',cols,...
    'Opacity',.5,'SmoothEdges',false, 'LineWidth',1);
%
% add acceptable lineage coexpression cell calls; top of circle lowest
% opal, bottom highest
%
% get top semicircle for shape
%
tx = radius * [cos(0:v:pi),cos(0)];
ty = radius * [sin(0:v:pi), sin(0)];
%
% get bottom semicircle for shape
%
bx = radius * [cos(pi:v:(2*pi)),cos(pi)];
by = radius * [sin(pi:v:(2*pi)),sin(pi)];
%
% create shape array for those phenotypes
%
tmarks = [];
bmarks = [];
%
% create color array for those phenotypes
%
tcols = [];
bcols = [];
%
for i1 = 1:length(Markers.add)
    curmark = Markers.add(i1);
    %
    % get top color or highest numeric opal in the coexpression
    %
    SW = cellfun(@(x)startsWith(curmark,x),Markers.all,'Uni',0);
    SW = [SW{:}];
    %
    tcol = mycol.all(2:end-1,:);
    tcol = tcol(SW,:);
    tcol = uint8(255 * tcol);
    %
    % get bottom color or lowest numeric opal in the coexpression
    %
    EW = cellfun(@(x)endsWith(curmark,x), Markers.all,'Uni',0);
    EW = [EW{:}];
    EWm = Markers.all{EW};
    %
    bcol = mycol.all(2:end-1,:);
    bcol = bcol(EW,:);
    bcol = uint8(255 * bcol);
    %
    % get cell X,Y Pos
    %
    ii = strcmp(tp2.Phenotype,curmark{1});
    x = tp2(ii,'CellXPos');
    y = tp2(ii, 'CellYPos');
    %
    x = table2array(x);
    y = table2array(y);
    %
    % make semicircle centered around the cell x and cell y positions
    %
    tx1 = repmat(tx,size(x,1),1);
    ty1 = repmat(ty,size(y,1),1);
    bx1 = repmat(bx,size(x,1),1);
    by1 = repmat(by,size(y,1),1);
    %
    x = repmat(x,1,size(tx1,2));
    y = repmat(y,1,size(ty1,2));
    %
    tx1 = tx1 + double(x);
    ty1 = ty1 + double(y);
    bx1 = bx1 + double(x);
    by1 = by1 + double(y);
    %
    % put the coordinates together
    %
    txy = [];
    bxy = [];
    %
    for i2 = 1:size(tx1,2)
        txy = [txy,tx1(:,i2),ty1(:,i2)];
        bxy = [bxy,bx1(:,i2),by1(:,i2)];
    end
    %
    % create shape array for those phenotypes
    %
    hh = size(txy,1);
    tmarks = [tmarks;txy];
    bmarks = [bmarks;bxy];
    %
    % create color array for those phenotypes
    %
    
    tcols = [tcols;repmat(tcol,hh,1)];
    bcols = [bcols;repmat(bcol,hh,1)];
end
%
imp = insertShape(imp,'FilledPolygon',tmarks,'Color',tcols,...
    'Opacity',.5,'SmoothEdges',false);
imp = insertShape(imp,'FilledPolygon',bmarks,'Color',bcols,...
    'Opacity',.5,'SmoothEdges',false);
%
% add expression marker line to the phenotype circles; r is offsets 
% for position of colored line
%
r = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5,...
    6, -6, 7, -7, 8, -8, 9, -9, 10, -10];
marks = [];
cols = [];
%
% separate the number into binary columns in order of Opals
%
% binary numbers = [1,2,4,8,16,32,64,128,256];
% Opals = [DAPI,480,520,540,570,620,650,690,780];
%
total_opals = [480,520,540,570,620,650,690,780]; % Opals without DAPI
%
t2 = tp2.ExprPhenotype;
phenb = [];
for i1 = 1:(length(total_opals)+1)
    t1 =  t2 ./ 2;
    t2 = floor(t1);
    t3 = t1 - t2;
    phenb(:,i1) = t3 ~= 0;
end
%
% remove DAPI column
%
phenb(:,1) = [];
%
% get extract the correct columns from phenb
%
ii = ismember(Markers.all,Markers.expr);
opals = Markers.Opals(ii);
colms = ismember(total_opals, opals);
phenb(:,~colms) = [];
%
% build the expression marker vectors for each cell
%
for i1 = 1:length(Markers.expr)
    curmark = Markers.expr{i1};
    %
    curcol = mycol.expr(i1,:);
    %
    % x and y positions of cells for current phenotype
    %
    ss = logical(phenb(:,i1));
    tp2.(lower(curmark)) = ss;
    x = tp2(ss,'CellXPos');
    y = tp2(ss, 'CellYPos');
    %
    x1 = table2array(x) + v2;
    x2 = table2array(x) - v2;
    %
    y = table2array(y) - r(i1);
    %
    xy = [x1 y x2 y];
    %
    % create shape array for those phenotypes
    %
    hh = size(xy, 1);
    marks = [marks;xy];
    %
    % create color array for those phenotypes
    %
    curcol = uint8(255 * curcol);
    cols = [cols;repmat(curcol,hh,1)];
end
%
% put the expression marker lines into the image
%
imp = insertShape(imp,'Line',marks,'Color',cols,...
    'Opacity',1,'SmoothEdges',false);
%
% rewrite legend over phenotypes
%
imp = insertText(imp,position,marksa,'BoxColor',[0,0,0],...
    'BoxOpacity',1,'FontSize',24,'TextColor', colsa);
%
% print image with just phenotypes on it
%
iname = [imageid.outfull,'cleaned_phenotype_image.tif'];
T = Tiff(iname,'w');
T.setTag(imageid.ds);
write(T,imp);
writeDirectory(T)
close(T)
%
% add segmentation
%
if doseg
    fullimages = reshape(fullimages,[],3);
    imp = reshape(imp,[],3);
    ss = reshape(simage,[],1);
    ss = find(ss>0);
    imp(ss,:) = repmat([255/.75 0 0],length(ss),1);
    fullimages(ss,:) = repmat([166 0 0],length(ss),1);
    %
    imp = reshape(imp,[imageid.size, 3]);
    fullimages = reshape(fullimages,[imageid.size, 3]);
    %
    % rewrite legend over segmentation
    %
    imp = insertText(imp,position,marksa,'BoxColor',[0,0,0],...
        'BoxOpacity',1,'FontSize',24,'TextColor', colsa);
    fullimages = insertText(fullimages,position,marksa,'BoxColor',[0,0,0],...
        'BoxOpacity',1,'FontSize',24,'TextColor', colsa);
    %
    iname = [imageid.outfull,'cleaned_phenotype_w_seg.tif'];
    T = Tiff(iname,'w');
    T.setTag(imageid.ds);
    write(T,imp);
    writeDirectory(T)
    close(T)
end
%
q.fig = tp2;
end
