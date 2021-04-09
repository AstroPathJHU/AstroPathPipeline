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
    Rfd, dRfd, MergeConfig, logstring)    
    QCImagesdate = [];
    QCImages = [];
    mergeroot = informpath;
    %
    % if QC folder does not exist but the Results folder does then run
    % Image function
    %
    QCfd = [informpath,'\Phenotyped\Results\QA_QC'];
    QCfl = dir([informpath,'\Phenotyped\Results\tmp_ForFiguresTables\*.mat']);
    %
    if exist(Rfd,'dir')
        Rfd = dir([Rfd,'\*table.csv']);
    else
        Rfd = [];
    end
    %
    if ~isempty(Rfd) && ~exist(QCfd,'dir') && ~isempty(QCfl)
        %
        % call image loop
        %
        CreateImageQAQC(mergeroot, sname, MergeConfig, logstring);
        %
    elseif ~isempty(Rfd) && ~exist(QCfd,'dir') && isempty(QCfl)
         QCImages = 0;
    end
    if exist(QCfd,'dir')
        %
        % create QC path and get images 
        %
        QCfd = [QCfd,'\Phenotype\All_Markers\*composite_image.tif'];
        QCfiles = dir(QCfd);
        %
        QCfl = dir([informpath,'\Phenotyped\Results\tmp_ForFiguresTables']);
        QCfl = QCfl(3:end);
        %
        % if QC output is empty and there are tmp_ForFigureTables then try
        % to make QC output again
        %
        if isempty(QCfiles) && ~isempty(QCfl)
            CreateImageQAQC(mergeroot, sname, MergeConfig, logstring);
            %
            QCfiles = dir(QCfd);
        end
        %
        if ~isempty(QCfiles) 
            %
            % get date of most recent file
            %
            [~,idx] = max([QCfiles(:).datenum]);
            dQCfd = QCfiles(idx).date;
            %
            % check if Results are newer than QC folder, if they are then
            % create a new QC output
            %
            QCfl = dir([informpath,'\Phenotyped\Results\tmp_ForFiguresTables']);
            QCfl = QCfl(3:end);
            %
            if datetime(dRfd) > datetime(dQCfd)
                if ~isempty(QCfl)
                    CreateImageQAQC(mergeroot, sname, MergeConfig, logstring);
                else 
                    rmdir(extractBefore(QCfd,'\QA_QC\'),'s');
                end
                %
                QCfiles = dir(QCfd);
                %
                % get a new date of most recent file created
                %
                [~,idx] = max([QCfiles(:).datenum]);
            end
            %
            % get number of files that QC was generated on
            %
            if ~isempty(QCfiles)
                QCImagesdate = QCfiles(idx).date(1:11);
                QCImages = length(QCfiles);
            end
        end
        %
    end
end