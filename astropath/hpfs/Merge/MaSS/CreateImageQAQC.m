%% CreateImageQAQC
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description:
%%% Create QC/QA Images for top 20 CD8 density fields of a case. Additional
%%% details located at the github repo: https://github.com/AstropathJHU/MaSS
%% --------------------------------------------------------------
%% Input:
%%% wd: working directory should contain the following subfolders
%%%     + Component_Tiffs
%%%     + Phenotyped
%%%     | - + ABX1
%%%     | - + ABX2
%%%     | - + ABXN... (a single folder for the antibody output from inForm)
%%%     | - + Results
%%%     | - - + Tables (the 'cleaned_phenotype_table.csv' files after merging by MaSS)
%%%     | - - + tmp_ForFiguresTables (.mat files created by MaSS corresponding to the
%%%         tables which meet the 600 cell \ 60 tumor cell criteria (if no tumor
%%%         marker only former criteria must be met)
%%% uc: sample name (only used in logging)
%%% MergeConfig: full path to the merge configuration file
%%% logstring: the intial portion of the logstring (project;cohort;)
%%% allimages: optional arguement; 0 (default) or #; do the image subset
%%%         or the whole image set
%%%
%% --------------------------------------------------------------
%% Usage:
%%% CreateImageQAQC(wd, uc, MergeConfig, logstring, [allimages])
%%% wd = '\\bki03\Clinical_Specimen_4\PZ1\inform_data'
%%% uc = 'PZ1';
%%% MergeConfig = '\\bki03\Clinical_Specimen_4\Batch\MergeConfig_01.xlsx'
%%% logstring = '1;2;'
%%% allimages = 0;
%%%
%% --------------------------------------------------------------
%% Output: 
%% described on the github repo: https://github.com/AstropathJHU/MaSS
%% --------------------------------------------------------------
%% License: 
% Copyright (c) 2019 Benjamin Green, Alex Szalay.
% Permission is hereby granted, free of charge, to any person obtaining a 
% copy of this software and associated documentation files (the "Software"),
% to deal in the Software without restriction, including without limitation 
% the rights to use, copy, modify, merge, publish, distribute, sublicense, 
% and/or sell copies of the Software, and to permit persons to whom the 
% Software is furnished to do so, subject to the following conditions:

% The above copyright notice and this permission notice shall be included 
% in all copies or substantial portions of the Software.

% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
% CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
% TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
% SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%% --------------------------------------------------------------
%%
function CreateImageQAQC(wd, uc, MergeConfig, logstring, allimages)
%
version = '0.01.0001';
if nargin < 4
    logstring = '';
end
%
doseg = 1; %option whether or not to generate images with segmentation maps
%
err_val = mywritetolog(wd, uc, logstring, '', 1, version);
e_code = err_handl(wd, uc, logstring, [], err_val);
if e_code == 1
    return
end
%
% get Markers structure
%
try
    err_str = 'parsing MergeConfig file';
    mywritetolog(wd, uc, logstring, err_str, 2);
    %    
    [Markers, err_val] = createmarks(MergeConfig);
    %
    e_code = err_handl(wd, uc, logstring, Markers, err_val);
    if e_code == 1
        return
    end
    %
catch
    err_val = 8;
    err_handl(wd, uc, logstring, [], err_val);
    return
end
%
%
% start the parpool if it is not open;
% attempt to open with local at max cores, if that does not work attempt
% to open with BG1 profile, otherwise parfor should open with default
%
if isempty(gcp('nocreate'))
    try
        numcores = feature('numcores');
        if numcores > 6
            numcores = 6; %#ok<NASGU>
        end
        evalc('parpool("local",numcores)');
    catch
        err_val = 10;
        e_code = err_handl(wd, uc, logstring, Markers, err_val);
        if e_code == 1
            return
        end
    end
end
%
% make the paths and select the hotspot charts
%
if nargin < 5
    allimages = 0;
end
%
try
    err_str = 'determining hotspot images';
    mywritetolog(wd, uc, logstring, err_str, 2);
    %
    [charts, err_val] = mkpaths(Markers, wd, allimages, doseg);
    %
    e_code = err_handl(wd, uc, logstring, Markers, err_val);
    if e_code == 1
        return
    end
    %
catch
    err_val = 11;
    err_handl(wd, uc, logstring, [], err_val);
    return
end
%
try 
    err_str = ['creating output for ', num2str(length(charts)),' fields'];
    mywritetolog(wd, uc, logstring, err_str, 2);
    %
    err_val = imageloop(wd, uc, logstring, Markers, charts, doseg);
    e_code = err_handl(wd, uc, logstring, Markers, err_val);
    if e_code == 1
        return
    end
    %
catch
    err_val = 14;
    err_handl(wd, uc, logstring, Markers, err_val);
    return
end
%
err_str = 'CreateQAQC finished';
mywritetolog(wd, uc, logstring, err_str, 2);
%
end
% function: mywritetolog
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 07/07/2020
%% --------------------------------------------------------------
%% Description
% create the log and output resultant error messages
%% --------------------------------------------------------------
%% input:
% err_val = exit code value indicating different errors
% loc = location in main code block of log file message
% wd = working directory of current specimen up to inform_data\Phenotyped
% tim = contains different file and time information
%%
function err_val = mywritetolog(wd, uc, logstring, err_str, locs, version)
%
tim = datestr(now,'yyyy-mm-dd HH:MM:SS');
logf = [wd,'\Phenotyped\Results\QA_QC\QA_QC.log'];
logp = [wd,'\Phenotyped\Results\QA_QC'];
err_val = 0;
%
if locs == 1
    %
    if exist(logp,'dir')
        try
            rmdir(logp,'s')
            mkdir(logp)
        catch
            err_val = 9;
            warning(['ERROR IN QA QC path:', wd,' ', uc]);
            return
        end
    else
        mkdir(logp)
    end
    %
    % create first line of file
    %
    str = [logstring, uc, ';CreateQAQC started-v',version,';', tim, '\r\n'];
    %
    fileID = fopen(logf,'wt');
    fprintf(fileID,str);
    fclose(fileID);
    %
end
%
% for error or warning messages write the message out in the correct format
%
if locs == 2
    %
    str = [logstring, uc, ';',err_str,';', tim, '\r\n'];
    %
    if isfile(logf)
        fileID = fopen(logf,'a');
    else 
        fileID = fopen(logf,'wt');
    end
    %
    fprintf(fileID,str);
    fclose(fileID);
    %
end
%
end
%% function: err_handl
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 07/07/2020
%% --------------------------------------------------------------
%% Description
% if there is an err_val that is not 0; output the corresponding message
% and return and exit code 1 to exit MaSS (errors) 0 to keep going
% (warnings)
%% --------------------------------------------------------
%%
function e_code = err_handl(wd, sname, logstring, Markers, err_val) %#ok<INUSL>
%
if err_val ~= 0
    %
    switch err_val
        case 1
            err_str = ['Error: MergeConfig file not ',...
                'found or could not be opened'];
            e_code = 1;
        case 2
            err_str = ['Warning: MergeConfig file read as double ',...
                'and two rows read as NaN. Changing first row to DAPI and ',...
                'ignoring the second'];
            e_code = 0;
        case 3
            err_str = ['Error: MergeConfig file "Opal" column could ',...
                'not be parsed'];
            e_code = 1;
        case 4
            err_str = ['Error: MergeConfig file "Coexpression" column could ',...
                'not be parsed'];
            e_code = 1;
        case 5
            err_str = ['Error: could not find DAPI row in the ',...
                'MergeConfig file'];
            e_code = 1;
        case 6
            err_str = ['Error: MergeConfig file should only contain ',...
                'one tumor designation'];
            e_code = 1;
        case 7
            err_str = ['Error: MaSS algorithm can only handle ',...
                'expression markers with multiple segmentation ',...
                'algorithms; check MergeCongif file'];
            e_code = 1;
        case 8
            err_str = ['Error: could not parse MergeConfig files'];
            e_code = 1;
        case 9
            err_str = ['Error: could not delete previous results folders.',...
                ' Please check folder permissions and that all ',...
                'files are closed'];
            e_code = 1;
        case 10
            err_str = ['Error: could not start par pool'];
            e_code = 1;
        case 11
            err_str = ['Error: could not find or open tmp_forfigurestables'];
            e_code = 1;
        case 12
            err_str = ['Error: cell seg summary.txt file ',...
                        'does not exist or is corrupt'];
            e_code = 1;
        case 13
            err_str = 'Error: check binary segmentation maps contain all 4 tissue+cell layers';
            e_code = 1;
        case 14
            err_str = ['Error: Could not export all QA QC images, check inForm tables'];
            e_code = 0;
        case 15
            err_str = 'Error: check inForm output files';
            e_code = 0;
    end
    %
    disp([sname, ';',err_str,';'])
    mywritetolog(wd, sname, logstring, err_str, 2);
    %
else
    e_code = 0;
end
end
%% imageloop
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%%  Loop through all images with proper error handling
%% --------------------------------------------------------------
%%
%% imageloop
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%%  Loop through all images with proper error handling
%% --------------------------------------------------------------
%%
function err_val = imageloop(wd, uc, logstring, Markers, charts, doseg)
%
err_val = 0;
e = cell(length(charts),1);
%
parfor i2 = 1:length(charts)
    log_name = extractBefore(charts(i2).name, '_cleaned');
    try 
        %
        %open mat lab data structure
        %
        err_str = ['CreateQAQC ', log_name, ' started'];
        mywritetolog(wd, uc, logstring, err_str, 2);
        %
        [s{i2}, imageid{i2}, mycol{i2}, imc, simage]...
            =  mkimageid(charts, i2, wd, Markers, doseg); %#ok<PFOUS,PFBNS>
        s{i2}.fig.CellXPos = round(s{i2}.fig.CellXPos);
        s{i2}.fig.CellYPos = round(s{i2}.fig.CellYPos);
        %
        %make overlayed phenotype map and save the data stucture for later
        %
        [s{i2},ima, imas] = mkphenim(s{i2}, Markers, mycol{i2},...
            imageid{i2}, imc, simage, doseg);
        %
        %make moscaics for expression and lineage markers
        %
        mkindvphenim(...
            s{i2}, mycol{i2}, imageid{i2},...
            imc, simage, Markers, ima, imas);
        mkexprim(...
            mycol{i2}, imageid{i2}, imc, Markers, ima, imas);
        %
        err_str = ['CreateQAQC ', log_name, ' finished'];
        mywritetolog(wd, uc, logstring, err_str, 2);
        e{i2} = 0;
    catch
        e{i2} = 1;
        err_str = ['ERROR: CreateQAQC ', log_name, ' failed'];
        mywritetolog(wd, uc, logstring, err_str, 2);
    end
    %
end
%
% make pie chart figure with heatmap
% in a separate loop because these figures are made via matlab graphs,
% which is not supported by parfor
%
%disp(str1{2});
%{
try
    %
    for i2 = 1:length(charts)
        %
        mkfigs(s{i2},Markers,imageid{i2}, mycol{i2});
        %
    end
    %
catch
    e{1} = ['Error in: ', uc, '; figure output failed for Image QA/QC'];
    disp(e{1})
    return
end
%}
%
% remove tmp_ForFiguresTables
%
poolobj = gcp('nocreate');
delete(poolobj);
%
if any(cell2mat(e))
    err_val = 14;
    return
else
    try
        rmdir([wd,'\Results\tmp_ForFiguresTables'],'s');
    catch
    end
end
%
end
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
colors = [eye(3); 1-eye(3); 1 1 1; 0 0 0;0.91 0.41 0.17; 1 .7529 .7961; ];
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
%% mkpaths
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% get the top 20 Immune fields from the .mat files in tmp_ForFigureTables
%%% and create the paths necessary for the rest of the code
%% --------------------------------------------------------------
%%
function [charts1, e] = mkpaths(Markers, wd, allimages, doseg)
%
e = 0;
%
% Remove any old ByImage directory
%
m{1} = [wd,'\Phenotyped\Results\QA_QC'];
%
% create new ByImage subdirectories
%
m{2} = [m{1},'\Phenotype'];
m{3} = [m{1},'\Tables_QA_QC'];
mkdir(m{3})
mkdir (m{2},'All_Markers')
%
m{1} = [m{1},'\Lin&Expr_Coex'];
%
for z = 1:length(Markers.all)
    mkdir(m{2}, Markers.all{z})
end
for z = 1:length(Markers.lin)
    mkdir(m{1}, Markers.lin{z})
end
for z = 1:length(Markers.add)
    mkdir(m{1}, Markers.add{z})
    mkdir(m{2}, Markers.add{z})
end
%
layers = length(Markers.Opals) + 2;
%
% get charts; determined based off of Blank, CD8, and Tumor percentages if
% there are more than 20 HPFs
%
try
    charts = dir([wd,'\Phenotyped\Results\tmp_ForFiguresTables\*.mat']);
catch
    e = 11;
    return
end
if isempty(charts)
    e = 11;
    return
end
%
charts1 = charts;
%
try 
    if length(charts1) > 20 && ~allimages
        inc = 1;
        while length(charts1) ~= 20 && inc <= 2
            formatspec = strcat(repmat('%s ',[1,5]),{' '},repmat('%f32 ',[1,10]),...
                { ' %s '},repmat('%f32 ',[1,5]),{' '},repmat('%f32 ',[1,5*layers]),...
                { ' %s '},repmat('%f32 ',[1,5]),{' '},repmat('%f32 ',[1,5*layers]),...
                { ' %s '},repmat('%f32 ',[1,5]),{' '},repmat('%f32 ',[1,5*layers]),...
                { ' %s '},repmat('%f32 ',[1,4]),{' '},repmat('%f32 ',[1,5*layers]),...
                {' '},repmat('%s ',[1,2]),{' '}, repmat('%f32 ',[1,4]),{' '}, ....
                repmat('%s ',[1,2]));
            formatspec = formatspec{1};
            %
            fd = [wd,'\Phenotyped\Results\tmp_ForFiguresTables'];
            nms = {charts(:).name};
            query3 = cell(1,length(charts));
            query2 = cell(1,length(charts));
            parfor i2 = 1:length(charts)
                nm = nms{i2};
                [query2{i2},query3{i2}] = delextrfields(fd,nm,wd,Markers,...
                    formatspec,inc);
            end
            %
            query2 = [query2{:}];
            query3 = [query3{:}];
            %
            query3 = query3(query2);
            charts1 = charts(query2);
            %
            [~,query4] = sort(query3,2,'descend');
            if length(query4) < 20
                a = query4(1:end);
            else
                a = query4(1:20);
            end
            %
            charts1 = charts1(a);
            inc = inc + .25;
        end
    end
catch
    e = 12;
    return
end
%
% check segmentation
%
if ~isempty(charts1) && doseg
    nm = extractBefore(charts(1).name,'cleaned_phenotype');
    %
    % get 1ry segmentation and see if it has proper layers
    %
    wd1 = [wd,'\Phenotyped\',Markers.seg{1},'\'];
    iname = [wd1,nm,'binary_seg_maps.tif'];
    props = imfinfo(iname);
    if length(props) < 4
        e = 13;
        return
    end
    %
    % check 2ry segmentations to see if they have proper layers
    %
    if ~isempty(Markers.altseg)
        for i1 = 1:length(Markers.altseg)
            mark = Markers.altseg{i1};
            wd1 = [wd,'\Phenotyped\',mark,'\'];
            iname = [wd1,nm,'binary_seg_maps.tif'];
            props = imfinfo(iname);
            if length(props) < 4
                e = 13;
                return
            end
        end
    end
end
%
end
%% delextrfields
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% Find list of which fields could be included
%% --------------------------------------------------------------
%%
function[query2, query3] = delextrfields(fd,nm,wd,Markers,formatspec,inc)
%
% get file name of orignal 1ry seg data
%
fname = [fd,'\',nm];
fid = extractBefore(nm,'cleaned_phenotype_table');
fname2 =  [wd,'\Phenotyped\',Markers.seg{1},'\',fid,'cell_seg_data_summary.txt'];
%
% read that table in
%
warning('off','MATLAB:table:ModifiedAndSavedVarnames')
s = readtable(fname2,'Format',formatspec,'Delimiter',...
    '\t','TreatAsEmpty',{' ','#N/A'});
%
% determine amount of Blank percentage
%
try
    ii1 = table2array(s(strcmp(s.TissueCategory,'Blank')&...
        strcmp(s.Phenotype,'All'),'TissueCategoryArea_pixels_'));
    ii2 = table2array(s(strcmp(s.TissueCategory,'All')&...
        strcmp(s.Phenotype,'All'),'TissueCategoryArea_pixels_'));
catch
    ii1 = table2array(s(strcmp(s.TissueCategory,'Blank')&...
        strcmp(s.Phenotype,'All'),'TissueCategoryArea_squareMicrons_'));
    ii2 = table2array(s(strcmp(s.TissueCategory,'All')&...
        strcmp(s.Phenotype,'All'),'TissueCategoryArea_squareMicrons_'));
end
%
% get a logical vector for those fields whose Blank tissue is less than
% (inc*.25*ii2) where inc grows on each iteration (to allow more blank
% space if there where not 20 fields that fit the previous criteria) ii2 is
% the amount of total tissue area on the slide
%
query2 = ii1 < (inc * .25 * ii2);
%
% check for the amount of immune inflitration via CD8+ population
%

s = load(fname);
s = s.fData;
query3 = height(s.fig(strcmp(s.fig.Phenotype,Markers.Immune{1}),:));
%
end
%% mkimageid function
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% creates variables for a single image
%% --------------------------------------------------------------
%%
function [q, imageid, mycol, imc, simage] =...
    mkimageid(charts, inum, wd, Markers, doseg)
%
% set image output properties
%
imageid.ds.Photometric = Tiff.Photometric.RGB;
imageid.ds.BitsPerSample   = 8;
imageid.ds.SamplesPerPixel = 3;
imageid.ds.SampleFormat = Tiff.SampleFormat.UInt;
imageid.ds.RowsPerStrip = 41;
imageid.ds.MaxSampleValue = 256;
imageid.ds.MinSampleValue = 0;
imageid.ds.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
imageid.ds.Software = 'MATLAB';
imageid.ds.ResolutionUnit = Tiff.ResolutionUnit.Inch;
imageid.ds.XResolution = 300;
imageid.ds.YResolution = 300;
imageid.ds.Compression = Tiff.Compression.LZW;
%
% get chart that correspond to inum
%
nc = [charts(inum).folder,'\',charts(inum).name];
q = load(nc);
q = q.fData;
q.fname = charts(inum);
q.fig.CellXPos = q.fig.CellXPos + 1;
q.fig.CellYPos = q.fig.CellYPos + 1;
%
% some image designations
%
imageid.wd = wd;
imageid.id = extractBefore(q.fname.name,'cleaned_phenotype_table.mat');
%
% write out Tables that comes from this image
%
writetable(q.fig,[wd,'\Phenotyped\Results\QA_QC\Tables_QA_QC\',...
    erase(q.fname.name,'.mat'),'.csv']);
%
% image input fname for segmentation images
%
sim{1} = [wd,'\Phenotyped\',Markers.seg{1},'\',imageid.id];
for i1 = 1: length(Markers.altseg)
    sim{i1+1} = [wd,'\Phenotyped\',Markers.altseg{i1},'\',imageid.id];
end
%
% image output fname for the full Marker images
%
imageid.outfull = [wd,...
    '\Phenotyped\Results\QA_QC\Phenotype\All_Markers\',imageid.id];
%
% image output fname for lineage markers
%
for i1 = 1:length(Markers.lin)
    imageid.outABlin{i1} = [wd,...
        '\Phenotyped\Results\QA_QC\Phenotype\',Markers.lin{i1},'\',imageid.id];
    imageid.outABcoex{i1} = [wd,'\Phenotyped\Results\QA_QC\Lin&Expr_Coex\',...
        Markers.lin{i1},'\',imageid.id];
end
%
% image output fname name for additional lineage markers (ie coexpression)
% image output fname for expression marker coexpression on lineage markers
%
for i2 = 1:length(Markers.add)
    imageid.outABlin{i1+1} = [wd,'\Phenotyped\Results\QA_QC\Phenotype\',...
        Markers.add{i2},'\',imageid.id];
    imageid.outABcoex{i1+1} = [wd,'\Phenotyped\Results\QA_QC\Lin&Expr_Coex\',...
        Markers.add{i2},'\',imageid.id];
    i1 = i1+1;
end
%
% image output fname for expression markers
%
for i1 = 1: length(Markers.expr)
    imageid.outABexpr{i1} = [wd,'\Phenotyped\Results\QA_QC\Phenotype\',...
        Markers.expr{i1},'\',imageid.id];    
end
ii = ismember(Markers.all, Markers.expr);
imageid.exprlayer = Markers.Opals(ii);
%
idx = find(Markers.nsegs > 1);
idx_count = length(imageid.outABexpr);
%
if idx
    for i1 = 1:length(idx)
        cidx = idx(i1);
        for i2 = 2:Markers.nsegs(cidx)
            idx_count = idx_count + 1;
            str = [wd,'\Phenotyped\Results\QA_QC\Phenotype\',...
                Markers.all{cidx},'_',num2str(i2)];
            if ~exist(str, 'dir')
                mkdir(str);
            end
            imageid.outABexpr{idx_count} = [str,'\',imageid.id];
            imageid.exprlayer = [imageid.exprlayer;Markers.Opals(cidx)];
        end
    end
end
%
% fname for the component_Tiff image
%
iname = [wd,'\Component_Tiffs\',...
    imageid.id,'component_data.tif'];
%
% read in all component images
%
props = imfinfo(iname);
imageid.size = [props(1).Height, props(1).Width];
%
imageid.ds.ImageLength = props(1).Height;
imageid.ds.ImageWidth = props(1).Width;
%
for i2 = 1:8
    if strcmp(props(i2).ColorType, 'grayscale')
        im(:,1) = reshape(imread(iname,i2),[],1);
        imc(:,i2) =(im(:,1)./max(im(:,1)));
    end
end
%
mycol.all = Markers.mycol.all;
%
% lineage marker colors only
%
lins = ismember(Markers.all,Markers.lin);
mycol.lin = mycol.all(2:end-1,:);
mycol.lin = mycol.lin(lins,:);
%
% expression marker colors only
%
expr = ismember(Markers.all,Markers.expr);
mycol.expr = mycol.all(2:end-1,:);
mycol.expr = mycol.expr(expr,:);
%
%%%segmentation images%%%
%
if doseg
    %
    % get rows from each alternative segmentation in the main table
    %
    trows = false(height(q.fig),length(Markers.altseg));
    for i1 = 1:length(Markers.altseg)
        trows(:,i1) = strcmp(q.fig.Phenotype,Markers.altseg{i1});
        cellnums = double(q.fig.CellNum(trows(:,i1)));
        %
        % read in alternative segmentations; this only works if there is 
        % tissue segmentation and nuclear segmentation in the 
        % binary_seg image; cytoplasm
        %
        s1 = imread([sim{i1 + 1},'binary_seg_maps.tif'], 4);
        %
        % set cell labels of segmentation image that are not 
        % in the main table to zero
        %
        s1(~ismember(double(s1),cellnums)) = 0;
        %
        s1 = reshape(s1,[],1);
        simage3{i1 + 1} = s1;
    end
    %
    % get every row for alternative segmentation in the main table
    %
    trowsall = sum(trows,2) > 0;
    %
    % read in primary segmentation image
    %
    s1 = imread([sim{1},'binary_seg_maps.tif'],4);
    %
    % get cellnums of primary segmentation data
    % (ie data not in any alt segs)
    %
    cellnums = double(q.fig.CellNum(~trowsall,:));
    %
    s1(~ismember(double(s1),cellnums))=0;
    s1 = reshape(s1,[],1);
    %
    simage3{1} = s1;
    %
    % read in tissue segmentation
    %
    % sum the images across the segmentations to create a single unique
    % segmentation
    %
    simage2 = [simage3{:}];
    %
    simage = sum(simage2,2);
    %
    simage(simage>0) = .5;
    %
    simage = reshape(double(simage), imageid.size);
else
    simage = zeros(imageid.size);
end
%
end
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
%%  mkindvphenim
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% This fucntion creates image mosiacs for lineage markers and dual
%%% expressions with expression markers
%% --------------------------------------------------------------
%%
function mkindvphenim(d, mycol, imageid, im,...
    ims, Markers, im_full_color, im_full_color_seg)
%
[Image,expr] = getset(Markers, imageid);
%
% get locations of segmentation
%
seg = reshape(ims,[],1);
seg = find(seg > 0);
scol = uint8(255 * .65);
%
for M = 1:length(Image.all_lineages)
    %
    % lineage image with dapi
    %
    current_marker = Image.all_lineages{M};
    %
    % for additional markers, aka dual lineage expression, we need to set
    % up special image and color vectors. Otherwise just use the specified
    % layer
    %
    if ismember(current_marker, Markers.add)
        %
        % get the position vectors
        %
        [SW, EW] = resolveMultiLin(current_marker, Markers);
        %
        % get the image columns and color
        %
        im_lineage_dapi = [im(:,1), im(:,EW), im(:,SW)];
        cc = [mycol.all(1,:); mycol.all(EW,:);  mycol.all(SW,:)];
        compartment = 'NULL';
        %
    else
        %
        % get the image columns and color
        %
        im_lineage_dapi =[im(:,1), im(:,Image.layer(M))];
        cc = [mycol.all(1,:); 1 1 1];
        compartment = Image.Compartment(M);
    end
    %
    % get the lineage image with and without dapi in color
    %
    [im_lineage_dapi_color, im_lineage_nodapi_color] = ...
        prepimages(im_lineage_dapi, cc, imageid.size, scol, seg);
    [im_lineage_dapi_color_noseg, im_lineage_nodapi_color_noseg] = ...
        prepimages(im_lineage_dapi, cc, imageid.size, scol, []);
    %
    % get the locations of the positive cells
    %
    ii = strcmp(d.fig.Phenotype,Image.all_lineages{M});
    data.ii = ii;
    data.pos = d.fig(ii,:);
    x = data.pos.CellXPos;
    y = data.pos.CellYPos;
    xy = [x y];
    data.xy = xy;
    %
    if height(data.pos) > 1
        %
        % create single color image for phenotyped image with dapi
        %
        create_color_images(im_lineage_dapi_color, imageid.outABlin{M},...
            Image,im_full_color, im_full_color_seg, data, d.fig, ...
            im_lineage_nodapi_color, im_lineage_dapi_color_noseg, ...
            im_lineage_nodapi_color_noseg, compartment)
        %
        % make images for expression marker & lineage coexpression
        %
        for t = 1:length(expr.namtypes)
            %
            % get the locations of the positive cells
            %
            ii = strcmp(d.fig.Phenotype,Image.all_lineages{M}) &...
                d.fig.(lower(expr.namtypes{t}));
            data.ii = ii;
            data.pos = d.fig(ii,:);
            xy = [data.pos.CellXPos data.pos.CellYPos];
            data.xy = xy;
            %
            if height(data.pos) > 1
                %
                % put image together for expr lin, dapi, segmentation
                %
                ly = expr.layers(t);
                ime = im(:,ly);
                imela = [im_lineage_dapi,ime];
                cc1 = [cc; mycol.all(ly,:)];
                %
                [imel, imelnd] = ...
                    prepimages(imela, cc1, imageid.size, scol, seg);
                [imel_noseg, imelnd_noseg] = ...
                    prepimages(imela, cc1, imageid.size, scol, []);
                %
                create_color_images(imel, [imageid.outABcoex{M},...
                    expr.namtypes{t}], Image, im_full_color,...
                    im_full_color_seg, data, d.fig,...
                    imelnd, imel_noseg, imelnd_noseg, compartment);
            end
        end
    end
end
%
%
end
%% getset
%% Created by: Benjamin Green
%% ---------------------------------------------
%% Description
% initialize the variables for the mkindvphenim funtion
%% ---------------------------------------------
%%
function [Image,expr] = getset(Markers,imageid)
%
Image.mossize = 50;
expr.namtypes = Markers.expr;
%
Image.ds = imageid.ds;
%
[a, ~] = ismember(Markers.all, expr.namtypes);
Image.ExprCompartment = Markers.Compartment(a);
a = find(a) + 1;
%
expr.layers = a;
%
Image.all_lineages = [Markers.lin, Markers.add];
%
[a, ~] = ismember(Markers.all, Image.all_lineages);
Image.Compartment = Markers.Compartment(a);
a = find(a) + 1;
%
Image.layer = a;
%
end
%% resolveMultiLin
%% Created by: Benjamin Green
%% ---------------------------------------------
%% Description
% determine the position vectors for multiple lineage
% coexpression
%% ---------------------------------------------
%%
function [SW, EW] = resolveMultiLin(current_marker, Markers)
%
% get top color or highest numeric opal in the coexpression
%
SW = cellfun(@(x)startsWith(current_marker,x),Markers.all,'Uni',0);
SW = [SW{:}];
SW = find(SW) + 1;
%
% get bottom color or lowest numeric opal in the coexpression
%
EW = cellfun(@(x)endsWith(current_marker,x), Markers.all,'Uni',0);
EW = [EW{:}];
EW = find(EW) + 1;
%
end
%% prepimages
%% Created by: Benjamin Green
%% ---------------------------------------------
%% Description
% prepare the images by multiplying by the corresponding color vectors and
% adding the segmentation.
%% Input
% im = image column vector where the first column is dapi
% c_map = color matrix for the corresponding image
% im_size = [h w] of the returned image
% scol = the shade of red for the segmentation map
% seg = the segmentation column vector
%% Ouput
% im_dapi = color image in matrix format with segmentation; with dapi
% im_nodapi = color image in matrix format with segmentation; no dapi
%% ---------------------------------------------
%%
function [im_dapi, im_nodapi] = prepimages(im, c_map, im_size, scol, seg)
%
% create dapi images first
%
im_dapi = 180 * sinh(1.5 * im) * c_map;
im_dapi(seg,:) = repmat([scol 0 0], length(seg),1);
im_dapi = uint8(im_dapi);
im_dapi = reshape(im_dapi,[im_size, 3]);
%
% create no dapi images next
%
im = im(:,2:end);
c_map = c_map(2:end,:);
im_nodapi = 180 * sinh(1.5 * im) * c_map;
if ~isempty(seg)
    im_nodapi(seg,:) = repmat([scol 0 0], length(seg),1);
end
im_nodapi = uint8(im_nodapi);
im_nodapi = reshape(im_nodapi,[im_size, 3]);
%
end
%% create_color_images
%% Created by: Benjamin Green
%% ---------------------------------------------
%% Description
% create 4 images for each input; single expression images for markers
% showing only positive colors, single expression images for markers
% showing all colors, image mosiacs for single expression images showing up
% to 25 + and 25 - cells -- both with and without DAPI
%% ---------------------------------------------
%%
function create_color_images(im, imageidout, Image,...
    im_full_color, im_full_color_seg, data, d, im_nodapi,...
    im_dapi_noseg, im_nodapi_noseg, compartment)
%
stypes = {'','_no_seg'};
if strcmp(compartment, 'Nucleus')
    dotcolor = 'green';
else
    dotcolor = 'white';
end
%
% get the data sample
%
data.neg = d(~data.ii,:);
if height(data.neg) > 25
    data.neg =  datasample(data.neg,25,1,'Replace',false);
end
if height(data.pos) > 25
    data.pos = datasample(data.pos,25,1,'Replace',false);
end
data.mos = cat(1,data.pos,data.neg);
%
for i1 = 1:2
    stype = stypes{i1};
    %
    if i1 == 1
        ims = im;
    else
        ims = im_dapi_noseg;
    end
    %
    Image.image = insertMarker(ims, data.xy,'+','color',dotcolor,'size',1);
    iname = [imageidout,'single_color_expression_image',stype,'.tif'];
    write_image(iname,Image.image,Image)
    %
    % Create the Image Mosiacs for local positive images with dapi
    %
    
    Image.x = data.mos.CellXPos;
    Image.y = data.mos.CellYPos;
    Image.imname = [imageidout,'cell_stamp_mosiacs_pos_neg',stype,'.tif'];
    makemosaics(Image)
    %
    if i1 == 1
        ims = im_nodapi;
    else
        ims = im_nodapi_noseg;
    end
    %
    % Create the Image Mosiacs for local positive images without dapi
    %
    Image.image = insertMarker(ims,data.xy,'+','color',dotcolor,'size',1);
    Image.imname = [imageidout,'cell_stamp_mosiacs_pos_neg_no_dapi',stype,'.tif'];
    makemosaics(Image)
    %
end
%
% create full color image for phenotyped image with dapi
%
imp = insertMarker(im_full_color_seg, data.xy, '+','color','white','size',1);
iname = [imageidout,'full_color_expression_image.tif'];
write_image(iname,imp,Image)
%
imp = insertMarker(im_full_color, data.xy, '+','color','white','size',1);
iname = [imageidout,'full_color_expression_image_no_seg.tif'];
write_image(iname,imp,Image)
%
end
%% write_image
%% Created by: Benjamin Green
%% ---------------------------------------------
%% Description
% write out the image 'Image' to the file iname using the Tiff library
%% ---------------------------------------------
%
%%
function write_image(iname,im,Image)
T = Tiff(iname,'w');
T.setTag(Image.ds);
write(T,im);
writeDirectory(T)
close(T)
end
%% makemosaics
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%%  make the cell stamp mosiac images
%% --------------------------------------------------------------
%%
function makemosaics(Image)
%%
%%%%%This function makes mosiacs to compare two ABs%%%%%%%%%%%%%%%%%%%%%%%%%
%
%image is a data structure with at least 5 fields
%.size - pixel size of cut outs
%.x & .y - x&y coordinates of cell/mosaic centers
%.image - the actual image to make the mosiac
%.imname - the path and file name
%
if ~isempty(Image.x)
    %
    %cut the image
    %
    xmins = Image.x - Image.mossize/2;
    ymins = Image.y - Image.mossize/2;
    h = repmat(Image.mossize, length(xmins),1);
    w = repmat(Image.mossize, length(xmins),1);
    box = [xmins ymins h w];
    mos = cell(length(xmins),1);
    %
    % find dimensions of the grid and number of blank fields
    %
    N = length(xmins);
    c = floor(sqrt(N));
    r = ceil(N/c);
    dims = (c * r);
    blnk = dims - N;
    blnk = blnk + N;
    %
    % get image cut outs
    %
    for i1 = 1:N
        m = imcrop(Image.image,box(i1,:));
        mos{i1,1} = imresize(m,...
            [Image.mossize + 1 Image.mossize + 1]);
    end
    %
    % add blank images
    %
    bb = zeros(Image.mossize + 1,Image.mossize + 1,3, 'uint8');
    for i1 = (N+1):blnk
        mos{i1,1} = bb;
    end
    %
    % make the mosaic and figure
    %
    mos = cell2mat(reshape(mos(:,:,:), r, c).');
    %
    rim = 2.5 * r * Image.mossize + 1;
    cim = 2.5 * c * Image.mossize + 1;
    %
    mos1 = imresize(mos,[cim, rim]);
    Image.ds.ImageLength = cim;
    Image.ds.ImageWidth = rim;
    %
    % print image
    %
    T = Tiff(Image.imname,'w');
    T.setTag(Image.ds);
    write(T,mos1);
    writeDirectory(T)
    close(T)
end
end

%%  mkexprim
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% This fucntion creates image mosiacs for expression markers
%% --------------------------------------------------------------
%%
function mkexprim(mycol, imageid, im,...
    Markers, im_full_color, im_full_color_seg)
%
[Image,~] = getset(Markers,imageid);
%
% get the segmentations for each expression variable, including dual 
% segmentations 
%
[ims, expr, d2] = getSegMaps(imageid, Markers);
%
for t = 1:length(expr.namtypes)
    %
    if (width(d2{t}) ~= 7)
        continue
    end
    %
    compartment = expr.compartment(t);
    %
    % put image together fused for expr, dapi, segmentation
    %
    ly = expr.layer(t);
    %
    imea = [im(:,1),im(:,ly)];
    cc = [mycol.all(1,:); 1 1 1];
    %
    seg = ims(:,t);
    seg = find(seg > 0);
    scol = uint8(255 * .65);
    % ---------------------------------------------------------------------
    % need to recreate segmentation vectors based on Markers.nsegs %%%%%%%
    % ---------------------------------------------------------------------

    [ime, imend] = ...
        prepimages(imea, cc, imageid.size, scol, seg);
    [ime_noseg, imend_noseg] = ...
        prepimages(imea, cc, imageid.size, scol, []);
    %
    % get the positive cells
    %
    ii = d2{t}.ExprPhenotype;
    %
    % set up struct
    %
    data.ii = ii;
    data.pos = d2{t}(ii,:);
    x = data.pos.CellXPos;
    y = data.pos.CellYPos;
    xy = [x y];
    data.xy = xy;
    %
    if height(data.pos) > 1
        %
        create_color_images(ime, imageid.outABexpr{t},...
            Image,im_full_color, im_full_color_seg, data, d2{t}, imend, ...
            ime_noseg, imend_noseg, compartment);
        %
    end
end
end
%%  getSegMaps
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
% This fucntion gets the segmentation maps for each expr variable
% including multiple segmentations. Returns a column vector with binary
% segmentation, a data struct with two fields (nametypes and layer), and
% modifies the outABexpr class of imageid. Also modifies the expression
% marker columns of the d.fig table to separate multiple segmentations.
%% Output
% ims = a column vector where each column is the binary map for the
% segmentation
% expr.namtypes = cell array of expression markers including additional
% segmentations for multiple segmentations on the same AB
% expr.layer = numeric array for the layer in the compenet tiff of each
% segmentation
% imageid.outABexpr = cell array of destination paths for the images
% d.fig = columns for multiple segmentations divided into parts
%% --------------------------------------------------------------
%%
function [ims, expr, d2] = getSegMaps(imageid, Markers)
%
% get antibody folder names
%
AB_fdnames = cellfun(@(x)extractBetween(x,'\Phenotype\',['\',imageid.id]),...
    imageid.outABexpr,'Uni',0);
AB_fdnames = [AB_fdnames{:}];
expr.namtypes = AB_fdnames;
expr.layer = imageid.exprlayer;
[~, loc] = ismember(expr.layer, Markers.Opals);
expr.compartment = Markers.Compartment(loc);
%
% get the cell x and y positions from each segmentation map
%
seg_types = [Markers.seg,Markers.altseg];
layers = length(Markers.Opals) + 2;
filnm = [imageid.id,'cell_seg_data'];
%
xy_seg = cellfun(@(x) get_pheno_xy(filnm,x,imageid.wd,layers),seg_types,'Uni',0);
xy_expr = cellfun(@(x) get_pheno_xy(filnm,x,imageid.wd,layers),AB_fdnames,'Uni',0);
d2 = xy_expr;
%
% convert to subscripts so we can perform matrix comparisons
%
loc_seg = cellfun(@(x) sub2ind(imageid.size,...
    x.CellYPosition,x.CellXPosition), xy_seg,'Uni',0);
num_seg = cellfun('length',loc_seg);
%
loc_expr = cellfun(@(x) sub2ind(imageid.size,...
    x.CellYPosition,x.CellXPosition), xy_expr,'Uni',0);
num_expr = cellfun('length',loc_expr);
%
% for each expression marker see which segmentation map it fits in
%
ims = zeros(imageid.size(1) * imageid.size(2),length(xy_expr));
%
for i1 = 1:length(xy_expr)
    %
    % find segmentations that have the same number of cells
    %
    idx = find(num_expr(i1) == num_seg); 
    %
    % if more than one segmentation type has the same number of cells
    % compare positions to determine current segmenation map
    %
    if length(idx) > 1
        for i2 = 1:length(idx)
            val = loc_expr{i1} == loc_seg{idx(i2)};
            if sum(val) == length(loc_expr{i1})
                c_seg = seg_types{idx(i2)};
                break
            end
        end
    elseif length(idx) == 0
        continue
    else
        c_seg = seg_types{idx};
    end
    %
    % read in that segmentation map and convert it to a column vector
    %
    folds = [imageid.wd,'\Phenotyped\',c_seg];
    im_name = [imageid.id,'binary_seg_maps.tif'];
    im_full = fullfile(folds,im_name);
    %
    seg_im = imread(im_full,4);
    ims(:,i1) = reshape(seg_im,[],1);
    %
    % make binary columns for d2
    %
    ii = expr.layer(i1) == Markers.Opals;
    AB = Markers.all(ii);
    expr.layer(i1) = find(ii) + 1;
    expr.compartment(i1) = Markers.Compartment(ii);
    d2{i1}.ExprPhenotype = strcmp(d2{i1}.Phenotype, AB);
    d2{i1}.CellXPos = d2{i1}.CellXPosition;
    d2{i1}.CellYPos = d2{i1}.CellYPosition;
end
%          
end
%% get_pheno_xy
%% Created by: Benjamin Green
%% ---------------------------------------------------------
%% Description 
% get the xy positions of each cell for a given phenotype marker
%% ---------------------------------------------------------
%%
function [xy] = get_pheno_xy(filnm,marker,wd1,layers)
%%-----------------------------------------------------------
%% load the csv file with the given marker and extract xy positions of each
%% cell
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
T = readtable([wd1,'\Phenotyped\',marker,'\',filnm,'.txt'],'Format',formatspec,...
    'Delimiter','\t','TreatAsEmpty',{' ','#N/A'});
vars = T.Properties.VariableNames;
xy = T(:,{'CellID','CellXPosition','CellYPosition','Phenotype'});
%
if any(contains(vars,'pixels'))
    xy.CellXPosition = xy.CellXPosition + 1;
    xy.CellYPosition = xy.CellYPosition + 1;
    return
end
%
filnm2 = extractBetween(filnm,'[',']');
filnm2 = strsplit(filnm2{1},',');
fx = str2double(filnm2{1});
fy = str2double(filnm2{2});
%
fold = [wd1,'\Component_Tiffs'];
iname = [fold,'\',replace(filnm,...
    'cell_seg_data','component_data.tif')];
imageinfo = imfinfo(iname);
W = imageinfo.Width;
H = imageinfo.Height;
scalea = 10^4 *(1/imageinfo(1).XResolution);
%
fx = (fx - scalea*(W/2)); %microns
fy = (fy - scalea*(H/2)); %microns
%
xy.CellXPosition = floor(1/scale .* (xy.CellXPosition - fx)) + 1;
xy.CellYPosition = floor(1/scale .* (xy.CellYPosition - fy)) + 1;
%
end
%% mkfigs
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% make the pie chart/ heatmap figures
%% --------------------------------------------------------------
%%
function mkfigs(d,Markers,imageid, mycol)
%
% This section will run for expression markers up to 15, after that the
% code will just skip this portion, if more expression markers are used one
% would need to either make a new page at every 15 or resize the pie
% charts at 'Positions'
%
if length(Markers.expr) > 15
    return
end
%
% create a color container to help specify colors in the pie charts; all
% additional markers added by acceptable coexpression will be black
%
for i1 = 1:length(Markers.lin)
    cmapvar{i1} = mycol.lin(i1,:);
end
%
for i2 = 1:length(Markers.add)
    i1 = i1 + 1;
    cmapvar{i1} = [0 0 0];
end
%
cmapvar{i1 + 1} = [0 0 1];
%
marks = [Markers.lin, Markers.add, 'Other'];
colorcont = containers.Map(marks, cmapvar);
%
% specify the positions for the pie charts:
%
% left shift = .18
% down shift = .32
% Grid:
% 1 - 3 - 7 - 10 - 13
% 2 - 4 - 8 - 11 - 14
% 5 - 6 - 9 - 12 - 15
%
Positions = {...
    [0.04 0.64 0.1 0.2], [0.04 0.35 0.1 0.2],...
    [0.22 0.64 0.1 0.2], [0.22 0.35 0.1 0.2],...
    [0.04 0.06 0.1 0.2], [0.22 0.06 0.1 0.2],...
    [0.40 0.64 0.1 0.2], [0.40 0.35 0.1 0.2], ...
    [0.40 0.06 0.1 0.2], [0.58 0.64 0.1 0.2], ...
    [0.58 0.35 0.1 0.2], [0.58 0.06 0.1 0.2],...
    [0.76 0.64 0.1 0.2], [0.76 0.35 0.1 0.2],...
    [0.76 0.06 0.1 0.2]};
%
% get data types into a conditional format that is easier to work with for
% the pie charts
%
data{1} = d.fig;
types{1} = ['All Markers (n = ',num2str(height(d.fig)),')'];
for i1 = 1:length(Markers.expr)
    m = Markers.expr{i1};
    l = lower(Markers.expr{i1});
    d.fig.(l) = logical(d.fig.(l));
    data {i1 + 1} = d.fig(d.fig.(l),:);
    hh = num2str(height(data{i1+1}));
    types{i1 + 1} = [m,' Expression (n = ',hh,')'];
end
%
% Create figure
%
XX = figure('visible' , 'off');
set(gcf, 'units','normalized','outerposition',[0 0 1 1],...
    'PaperUnits','inches');
XX.PaperPositionMode = 'auto';
%
% add a title to the figure
%
str = extractBefore(d.fname.name,'_cleaned');
str = ['Image QC Output for: ', str];
annotation('textbox', [.004 .8 .1 .2], 'String',str,'FontSize',14,...
    'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on',...
    'Interpreter','none')
%
% Make pie charts: on the matlab help page - Label Pie Chart With Text and
% Percentages - can also provide additional changes to axis
%
for i2 = 1:length(data)
    %
    % get Phenotypes in this data type
    %
    m = unique(data{i2}.Phenotype);
    explode = ones(length(m),1)';
    %
    if ~isempty(m)
        %
        % get colors for markers in this data type
        %
        cmap = zeros(length(m), 3);
        %
        for i1 = 1:length(m)
            m1 = m{i1};
            c = colorcont(m1);
            cmap(i1,:) = c;
        end
        %
        % create graph at Position designated
        %
        p(i2) = axes('Position',Positions{i2});
        %
        pie(p(i2), categorical(string(data{i2}.Phenotype(:))),...
            explode);
        %
        title(p(i2), types{i2})
        %
        set(get(gca,'title'),'Position',[0.1 1.5 1])
        %
        colormap(p(i2),cmap);
    end
end
%
% Make a heat map
%
% create a color scale
%
r = [ones(1,30)', linspace(1,0,30)', linspace(1,0,30)'];
b = [linspace(0,1,30)', linspace(0,1,30)',ones(1,30)'];
cmap = vertcat(b,r);
%
% get number of unique phenotypes to make heatmap bars for
%
m = unique(d.fig.Phenotype);
marks = [Markers.all,Markers.add,'Other'];
m1 = ismember(marks,m);
m = marks(m1);
%
% set positions for the heatmap; the rest of the heatmap should shift
% appropriately if these values are changed
%
% ls: left side of the first heatmap on the chart
% szw: the width of the heatmap bars
% szh: the height of the heatmap bars
% b: the bottom of the heatmap
%
ls = .84;
szw = .075;
szh = .75;
b = .13;
%
% a check that will separate the figures into different image files if the
% data types need this
%
if (length(data) > 6 && length(m) > 3) ||...
        length(data) > 9 || ...
        (length(m) > 6 && length(data)~=1)
    print(XX, strcat(imageid.outfull,...
        'cleaned_phenotype_pie_charts.tif'),'-dtiff','-r0');
    close all
    XX = figure('visible' , 'off');
    set(gcf, 'units','normalized','outerposition',[0 0 1 1],...
        'PaperUnits','inches');
    XX.PaperPositionMode = 'auto';
    %
    % add title to figure
    %
    annotation('textbox', [.004 .8 .1 .2], 'String',str,'FontSize',14,...
        'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on',...
        'Interpreter','none')
    %
    % move the heatmap to the middle of the page
    %
    ls = .84 - 2 * szw;
    szw = .075;
    szh = .75;
    b = .13;
    %
end
%
% set the positions of the heatmap bars for each phenotype
%
Pos1 = [ls 0.13 szw szh];
%
for i1 = 1: length(m)
    ps1 = Pos1(i1,1) - szw;
    Pos1 = [Pos1;ps1 b szw szh];
end
Pos1 = flip(Pos1);
%
% create the heatmaps
% done in reverse order to help with formatting
%
for i1 = length(m):-1:1
    x1 = d.fig(strcmp(d.fig.Phenotype,m{i1}),:);
    %
    %find log intensity of each marker in total cell
    %
    x = [];
    for i2 = 1:length(Markers.Opals)
        intes = ['MeanEntireCell',num2str(Markers.Opals(i2))];
        var = (log(x1.(intes) + .001));
        x(i2,:) = var;
    end
    x = flip(x);
    %
    axes('Position',Pos1(i1,:));
    h(i1) = heatmap(x, 'Colormap', cmap,'ColorbarVisible', 'off',...
        'GridVisible', 'off');
    % h(i1).XLabel = m{i1};
end
%
% set the colorbar
%
h(length(m)).ColorbarVisible = 'on';
%
% set the y axis to the Opal names
%
Op = flip(Markers.Opals);
Op = num2str(Op);
Op = [repmat('Opal ',length(Op),1), Op];
Op = cellstr(Op);
h(1).YData = Op ;
%
% hide the cell numbers that show on the bottom of the heatmap
%
str = ' ';
ps(1,1) = Pos1(1,1) - .03;
ps(1,2) = b - .08;
ps(1,3) = length(m) * szw + .06;
ps(1,4) = b - ps(1,2) - .002;
%
annotation('textbox', ps, 'String',str,...
    'BackgroundColor' , 'white','EdgeColor','white');
%
% set the x axis to the marker names
%
for i1 = 1: length(m)
    m1 = m{i1};
    %[0.465 0.13 0.075 0.75]
    ps =  Pos1(i1,:);
    ps(1,1) = Pos1(i1,1) + szw/5;
    ps(1,2) = b - .04;
    ps(1,3) = .05;
    ps(1,4) = .03;
    annotation('textbox', ps, 'String', m1,'FontSize',14,...
        'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on');
end
%
% display which Opals target which ABs under the xaxis labels
%
Op = flip(Op);
Op1 = [];
for i1 = 1:length(Markers.all)
    A = Op{i1}(~isspace(Op{i1}));
    Op1 = [Op1,Markers.all{i1},': ',A,'; ',];
end
%
Op1 = ['Note: ',Op1];
%
ps(1,1) = Pos1(end,1) - (length(Markers.Opals) * .05);
%
%ps(1,1) = Pos1(1,1) - .05;
ps(1,2) = .055;
annotation('textbox', ps, 'String', Op1,'FontSize',14,...
    'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on');
%
% make a title for the heatmap
%
str = 'Heatmap of Phenotype vs Opal Intensities in Cells';
if length(m) <= 2
    ps(1,1) = Pos1(1,1) - szw/2;
else
    sz = floor((length(m)-1)/2);
    ps(1,1) = Pos1(sz,1) + szw/5;
end
%
ps(1,2) = Pos1(1,2) + szh + .01;
annotation('textbox', ps, 'String', str,'FontSize',16,...
    'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on');
%
print(XX, strcat(imageid.outfull,...
    'cleaned_phenotype_data.tif'),'-dtiff','-r0');
close all
%
end