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
