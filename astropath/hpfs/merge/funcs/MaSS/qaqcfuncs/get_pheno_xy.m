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
