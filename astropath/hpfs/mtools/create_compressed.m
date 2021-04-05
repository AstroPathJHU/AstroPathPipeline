%% create_compressed
%% Created by: Benjamin Green 12/27/2019
%% -------------------------------------------------
%% Description
% Script to compress the tables and component tiffs for a clinical specimen
% directory. Takes a main path and compressed path as input and creates a
% compress copy of the component tiffs with segmentation data and the
% tables. The compressed copies will be in a zipped folder with a name
% corresponding to either Component_Tiffs_X or Tables_X located in the
% corresponding location of the file structure. The '_X' indicates the
% saved file version starting at '_1'.
%% ----------------------------------------------------
%% Usage
%main_path = '\\bki04\Clinical_Specimen';
%compressed_path = '\\bki03\Compressed_Clinical_Specimens\Clinical_Specimen';
%% ---------------------------------------------------
%%
function [] = create_compressed(main_path,compressed_path)
%
fdnames = dir(main_path);
for i1 = 1:length(fdnames)
    %
    % main directory
    %
    wd1 = [main_path,'\',fdnames(i1).name,'\inform_data'];
    if ~exist(wd1,'dir')
        continue
    end
    %
    % component images
    %
    wd_components = [wd1,'\Component_Tiffs\*w_seg.tif'];
    %
    % path to zip for component images
    %
    wd2 = [compressed_path,'\',fdnames(i1).name,'\inform_data\Component_Tiffs'];
    %
    if ~exist(wd2,'dir')
        mkdir(wd2)
    end
    %
    % get zipped name for component images and zip to main directory -
    % limits strain on network
    %
    previous_zip_value = dir(wd2);
    current_zip_value = length(previous_zip_value) - 1;
    current_zip_name = [wd1,'\Component_Tiffs_',num2str(current_zip_value),'.zip'];
    %{
    % zipp component images 
    %
    tic
    try
        zip(current_zip_name,wd_components);
    catch
        fprintf(['Error in ',fdnames(i1).name,' component zipp \r']);
    end
    %
    new_location =  [wd2,'\Component_Tiffs_',num2str(current_zip_value),'.zip'];
    %
    % move to new location
    %
    try
        movefile(current_zip_name,new_location)
    catch
        fprintf(['Error in moving ',fdnames(i1).name,' component zipp \r']);
    end
    tt = toc;
    %
    fprintf([fdnames(i1).name,...
        ' component zipp; Elapsed time is ',num2str(tt),' seconds\r']);
    %}
    % path to tables
    %
    wd_tables = [wd1,...
        '\Phenotyped\Results\Tables\*_cleaned_phenotype_table.csv'];
    %
    % get zipped name for tables and zip to main directory - limits strain
    % on network
    %
    wd2 = [compressed_path,'\',fdnames(i1).name,...
        '\inform_data\Phenotyped\Results\Tables'];
    %
    if ~exist(wd2,'dir')
        mkdir(wd2)
    end
    current_zip_name = [wd1,'\Phenotyped\Results\Tables_1.zip'];
    new_location = [wd2,'\Tables_1.zip'];
    %
    % zipp tables 
    %{
    tic
    try
       % zip(current_zip_name,wd_tables);
    catch
       % fprintf(['Error in ',fdnames(i1).name,' tables zipp \r']);
    end
    %}
    % move to new location
    %
    try
        movefile(current_zip_name,new_location)
    catch
        fprintf(['Error in moving ',fdnames(i1).name,' tables zipp \r']);
    end
    %tt = toc;
    fprintf([fdnames(i1).name,...
         ' tables zipp; Elapsed time is seconds\r']);
end
end