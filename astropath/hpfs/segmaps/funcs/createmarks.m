%% createmarks
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% function takes in a folder location and creates the Markers data
%%% structure
%% --------------------------------------------------------------
%%
function [Markers, err_val] = createmarks(MergeConfig)
%
warning('off','MATLAB:table:ModifiedAndSavedVarnames')
Markers = [];
%
try
    BIDtbl = readtable(MergeConfig);
    B = BIDtbl(:,{'Opal','Target',...
    'TargetType','CoexpressionStatus','SegmentationStatus',...
    'SegmentationHierarchy', 'ImageQA','Compartment',...
    'NumberofSegmentations', 'Colors'});
catch
    err_val = 1;
    return
end
%
% check table input variables
%
[err_val, mycol] = extractcolorvec(B);
if err_val == 3
    err_val = 8;
    return
end
%
[B, err_val] = checkTableVars(B);
if ~ismember(err_val, [0,2])
    return
end
%
% start setting up Markers struct
%
Markers.Opals = B.Opal;
Markers.all = B.Target;
Markers.mycol = mycol;
Markers.Compartment = B.Compartment;
%
ii = strcmp('Tumor',B.ImageQA);
Markers.all_original = Markers.all;
%
% change Tumor marker designation to 'Tumor'
%
if sum(ii) == 1
    Markers.all(ii) = {'Tumor'};
    Markers.Tumor{1} = 'Tumor';
elseif sum(ii) > 1
    err_val = 6;
    return
else
     Markers.Tumor{1} = '';
end
%
ii = strcmp('Immune',B.ImageQA);
%
if sum(ii) == 1
    Markers.Immune = Markers.all(ii);
else
    Markers.Immune = Markers.all(find(ii, 1));
    err_val = 7;
end
%
% get lineage and expression markers
%
LT = strcmp(B.TargetType,'Lineage');
Markers.lin = Markers.all(LT);
%
ET = strcmp(B.TargetType,'Expression');
Markers.expr = Markers.all(ET);
%
% get the markers with multiple segmentations, this will only be a
% capability on expression markers
%
nsegs = B.NumberofSegmentations;
if iscell(nsegs)
    nsegs = cellfun(@(x) str2double(x), nsegs, 'Uni',0);
    nsegs = cell2mat(nsegs);
end
if find(nsegs(~ET) > 1)
    err_val = 7;
    return
end
Markers.nsegs = nsegs;
%
% Set up segmentation status to define number of segmentations and which is
% the primary segmentation
%
SS = B.SegmentationStatus;
Markers.SegStatus = SS;
%
ii = nsegs == 1 & ~ismember(Markers.all,Markers.expr);
SS = SS(ii);
mn = Markers.all(ii);
%
% get number of different segmentations, remove markers with multiple
% segmentations from the contention
%
[~,y,~] = unique(SS);
ii = y(1);
%
Markers.seg = mn(ii);
%
Markers.altseg = cell(length(y)-1,1);
for i1 = 2:length(y)
    ii = y(i1);
    Markers.altseg(i1-1) = mn(ii);
end
%
% get coexpression status for lineage markers
%
CS = B.CoexpressionStatus(LT);
ii = ~strcmp(CS,'NA') | ~strcmp(CS,'NaN');
CS = CS(ii);
%
% track the corresponding target
%
TCS = Markers.lin(ii);
%
% get segmentation heirarchy
%
SH = B.SegmentationHierarchy;
Markers.SegHie = SH(LT);
%
% CS that is not NA in lineage markers; find which coexpressions are
% acceptable
%
Markers.add = [];
sego = [];
for i1 = 1:length(CS)
    %
    % get current target and opal
    %
    T = TCS{i1};
    ii = strcmp(T,Markers.all);
    o = Markers.Opals(ii);
    o = o{1};
    %
    % check them against rest of targets in coexpression
    %
    CStest = CS(~strcmp(TCS,T));
    TCStest = TCS(~strcmp(TCS,T));
    %
    for i2 = 1:length(CStest)
        o1 = CStest{i2};
        T1 = TCStest{i2};
        %
        % if the current target matches one of the targets in the rest 
        %
        if contains(o1,o)
            %
            % if the Markers.add is not empty; are both markers already
            % contained together
            %
            if ~isempty(Markers.add) && sum(contains(Markers.add,T)...
                & contains(Markers.add,T1))
                continue
            else
                track = length(Markers.add) + 1;
                Markers.add{track} = [T1,T];
                ii = strcmp(T1, Markers.lin);
                seg1 = Markers.SegHie(ii);
                ii = strcmp(T, Markers.lin);
                seg2 = Markers.SegHie(ii);
                %
                seg = max([str2double(seg1{1}),str2double(seg2{1})]);
                sego{track} = num2str(seg);
            end
        end
    end
end
%
Markers.SegHie = [Markers.SegHie;sego'];
%
% get coexpression status for expression markers
%
CS = B.CoexpressionStatus(ET);
for i1 = 1:length(CS)
    T = CS{i1};
    T = reshape(T,3,[])';
    [s,~] = size(T);
    x = arrayfun(@(x)contains(Markers.Opals,T(x,:)),1:s,'Uni',0);
    x = horzcat(x{:});
    Markers.Coex{i1} = sum(x,2);
end
%
% reformat for proper dims
%
Markers.Opals = cellfun(@str2double, Markers.Opals, 'Uni',0);
Markers.Opals = cell2mat(Markers.Opals);
Markers.all = Markers.all';
Markers.all_original = Markers.all_original';
Markers.lin = Markers.lin';
Markers.expr = Markers.expr';
Markers.nsegs = Markers.nsegs';
Markers.seg = Markers.seg';
Markers.altseg = Markers.altseg';
Markers.SegHie = Markers.SegHie';
%
end
%% extractcolorvec
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% extracts the color vector from the Colors column in the merge config 
%%% file
%% --------------------------------------------------------------
%%
function[err_val, mycol] = extractcolorvec(B)
%
err_val = 0;
%
% colors
%
if height(B) <= 7
    %%blue%green%yellow%red%orange%cyan%magenta%black%%
    mycolab = [0 1 0;
        1 1 0;
        1 0 0;
        0.91 0.41 0.17;
        0 1 1;
        1 0 1;];
    mycol.all = [0 0 1;
        mycolab(1:height(B)-1, :);
        0 0 0];
    %
elseif height(B) <= 10 && height(B) > 7
    %%blue%coral%green%yellow%red%orange%cyan%magenta%white%black%%
    mycolab = [1 .7529 .7961;
        0 1 0;
        1 1 0;
        1 0 0;
        0.91 0.41 0.17;
        0 1 1;
        1 0 1;
        1 1 1;];
    mycol.all = [0 0 1;
        mycolab(1:height(B)-1, :);
        0 0 0];
    %
else
    mycol.all = [];
    err_val = 1;
    %{
    error(['Error in ImageLoop > mkimageid: \n'...
        'Need to add color values for ',n,...
        ' color panel to mkimageid function'], class(n));
    %}
end
%
color_names = {'red','green','blue','cyan', ...
    'magenta','yellow','white','black','orange','coral'};
color_names2 = {'r','g','b','c','m','y','w','k','o','l'};
colors = [eye(3); 1-eye(3); 1 1 1; 0 0 0; 1 .7529 .7961; 0.91 0.41 0.17;];
%
if isa(B.Colors, 'cell')
    [ii,loc] = ismember(B.Colors, color_names);
    [ii1,loc1] = ismember(B.Colors, color_names2);
    ii = ii + ii1;
    loc = loc + loc1;
    %
    if sum(ii) ~= length(B.Colors)
        new_colors = B.Colors(~ii);
        new_colors = replace(new_colors, {'[',']',' '},'');
        new_colors = cellfun(@(x) strsplit(x, ','), new_colors, 'Uni', 0);
        new_colors = cellfun(@str2double, new_colors, 'Uni', 0);
        if any(cellfun(@length, new_colors)~=3)
            err_val = err_val + 2;
            return
        end
        new_colors = cell2mat(new_colors);
        if any(new_colors > 255)
            err_val = err_val + 2;
            return
        end
        loc(~ii) = (length(colors) + 1):...
            ((length(colors)) + (length(B.Colors) - sum(ii)));
        colors = [colors;new_colors];
    end
    %
    mycol.all = [colors(loc,:); 0, 0, 0];
    %
else
    err_val = err_val + 2;
    return
end
%
end
%% function: checkTableVars
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 07/07/2020
%% --------------------------------------------------------------
%% Description
% checks that the table input is correct 
%% --------------------------------------------------------------
%%
function [B, err_val] = checkTableVars(B)
%%
% check the table variables to be sure they are in the correct format for
% the code. If they are not convert them.
%%
%
err_val = 0;
%
% check the data type for Opal column
%
if isa(B.Opal,'double')
   %
   % if B.Opal is a 'double' convert to a string 
   %
   tmpopal = num2cell(B.Opal);
   tmpopal = cellfun(@(x) num2str(x), tmpopal, 'Uni', 0);
   ii = strcmp(tmpopal, 'NaN');
   %
   if sum(ii) > 1
      err_val = 2;
      ii = find(ii,1);
   end
   %
   tmpopal(ii) = {'DAPI'};
   ss = size(tmpopal);
   if ss(1) == 1
       B.Opal = tmpopal';
   else
       B.Opal = tmpopal;
   end
end
%
if ~isa(B.Opal, 'cell')
  err_val = 3;
  return
end
%
% check the data type for the coexpression status column
%
if isa(B.CoexpressionStatus,'double')
   %
   % if B.Opal is a 'double' convert to a string 
   %
   tmpCS = num2cell(B.CoexpressionStatus);
   tmpCS = cellfun(@(x) num2str(x), tmpCS, 'Uni', 0);
   %
   for i1 = 1:length(tmpCS)
       tmpCS_n = tmpCS{i1};
       if length(tmpCS_n) > 3
           ii = 3:3:length(tmpCS_n) - 1;
           t(1:length(tmpCS_n)) = char(0);
           t(ii) = ',';
           tmpCS_n = [tmpCS_n;t];
           tmpCS_n = reshape(tmpCS_n(tmpCS_n ~= 0),1,[]);
           tmpCS{i1} = tmpCS_n;
       end
   end
   %
   B.CoexpressionStatus = tmpCS;
   %
end
%
B.CoexpressionStatus = cellfun(@(x) replace(x, ',',''),...
      B.CoexpressionStatus, 'Uni',0);
%
if ~isa(B.Opal, 'cell')
    err_val = 4;
end
%
% remove the DAPI row
%
dr = strcmp(B.Opal, 'DAPI');
if sum(dr) ~= 1
    err_val = 5;
end
B(dr,:) = [];
%
% check the last 3 columns are all set as numeric
%
SS = B.SegmentationStatus;
if iscell(SS)
    %SS = cell2mat(SS);
    B.SegmentationStatus = str2double(SS);
end
%
SH = B.SegmentationHierarchy;
if ~iscell(SH)
    SH = num2str(SH);
    SH = cellstr(SH);
    B.SegmentationHierarchy = SH;
end
%
SS = B.NumberofSegmentations;
if iscell(SS)
    %SS = cell2mat(SS);
    B.NumberofSegmentations = str2double(SS);
end
%
if ~iscell(B.Compartment)
    err_val = 8;
    return
end
%
end
