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
filepath = fileparts(mfilename('fullpath'));
addpath(genpath(filepath))
%
version = '0.01.0001';
if nargin < 4
    logstring = '';
end
%
doseg = 1; %option whether or not to generate images with segmentation maps
%
err_val = mywritetolog(wd, uc, logstring, '', 1, 'QA_QC', version);
e_code = err_handl(wd, uc, logstring, [], err_val, 'QA_QC');
if e_code == 1
    return
end
%
err_str = ['-wd: ', replace(wd, '\', '\\'), ' -sname: ', uc];
mywritetolog(wd, uc, logstring, err_str, 2, 'QA_QC');
%
% get Markers structure
%
err_str = ['parsing MergeConfig file: ', replace(MergeConfig, '\', '\\')];
mywritetolog(wd, uc, logstring, err_str, 2, 'QA_QC');
%
% get Markers structure
%
try
    %    
    [Markers, err_val] = createmarks(MergeConfig);
    %
catch
    err_val = 8;
    err_handl(wd, uc, logstring, [], err_val, 'QA_QC');
    return
end
%
e_code = err_handl(wd, uc, logstring, Markers, err_val, 'QA_QC');
if e_code == 1
    return
end
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
        e_code = err_handl(wd, uc, logstring, Markers, err_val, 'QA_QC');
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
err_str = 'determining hotspot images';
mywritetolog(wd, uc, logstring, err_str, 2, 'QA_QC');
%
try
    %
    [charts, err_val] = mkpaths(Markers, wd, allimages, doseg);
    %
catch
    err_val = 11;
    err_handl(wd, uc, logstring, [], err_val, 'QA_QC');
    return
end
%
e_code = err_handl(wd, uc, logstring, Markers, err_val, 'QA_QC');
if e_code == 1
    return
end
%
err_str = ['creating output for ', num2str(length(charts)),' fields'];
mywritetolog(wd, uc, logstring, err_str, 2, 'QA_QC');
err_val = imageloop(wd, uc, logstring, Markers, charts, doseg);

%
try 
    %
    err_val = imageloop(wd, uc, logstring, Markers, charts, doseg);
    %
catch
    err_val = 14;
    err_handl(wd, uc, logstring, Markers, err_val, 'QA_QC');
    return
end
%
e_code = err_handl(wd, uc, logstring, Markers, err_val, 'QA_QC');
if e_code == 1
    return
end
%
err_str = 'CreateQAQC finished';
mywritetolog(wd, uc, logstring, err_str, 2, 'QA_QC');
%
end


