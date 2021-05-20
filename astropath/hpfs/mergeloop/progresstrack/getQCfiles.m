%% getQCfiles
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% runs the QA_QC code for a specimen if it has not been run before or if
%%% there is newer results tables,
%%% it will also track the number of QA_QC files and QA_QC ready dates
%% --------------------------------------------------------------------
%%
function [QCImagesdate, QCImages]  = getQCfiles(sname, informpath,...
    Rfd, MergeConfig, logstring)
QCImagesdate = [];
QCImages = [];
%
% get the qc files and the tmp files
%
QCfd = [informpath,'\Phenotyped\Results\QA_QC'];
QCfls= dir([QCfd,'\Phenotype\All_Markers\*composite_image.tif']);
QCtmp = dir([informpath,'\Phenotyped\Results\tmp_ForFiguresTables\*.mat']);
Rfls = dir([Rfd,'\*table.csv']);
%
dRfd = return_date_max(Rfls);
dQCfls = return_date_max(QCfls);
dQCtmp = return_date_max(QCtmp);
%
% qc folder does not exist and tmp files are not empty create
%
for i1 = 1:3
    %
    if (~exist(QCfd,'dir') && ~isempty(dQCtmp)) || ...
            (isempty(QCfls) && ~isempty(dQCtmp)) || ...
            (~isempty(QCfls) && ~isempty(dQCtmp) ...
            && datetime(dQCfls) < datetime(dQCtmp))
        CreateImageQAQC(informpath, sname, MergeConfig, logstring);
    elseif (exist(QCfd,'dir') && isempty(dRfd)) || ...
            (exist(QCfd,'dir') && datetime(dRfd) > datetime(dQCfls))
        rmdir(QCfd,'s')
    end
    %
    % for the loop to triple verify imgaes
    %
    QCfls= dir([QCfd,'\Phenotype\All_Markers\*composite_image.tif']);
    QCtmp = dir([informpath,'\Phenotyped\Results\tmp_ForFiguresTables\*.mat']);
    Rfls = dir([Rfd,'\*table.csv']);
    %
    dRfd = return_date_max(Rfls);
    dQCfls = return_date_max(QCfls);
    dQCtmp = return_date_max(QCtmp);
    %
end
%
if ~isempty(dQCfls)
    QCImagesdate = dQCfls(1:11);
    QCImages = length(QCfls);
else
    QCImagesdate = [];
    QCImages = 0;
end
%
end