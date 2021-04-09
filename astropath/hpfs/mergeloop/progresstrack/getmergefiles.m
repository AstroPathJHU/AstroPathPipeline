%% getmergefiles
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% runs the merge code for a specimen if it has not been run before or if
%%% there is newer inform data in one of the ABx folders that the results,
%%% it will also track the number of merged files and merge dates
%% --------------------------------------------------------------------
%%
function [MergeTbls, MergeTblsDate, Rfd, dRfd] = ...
    getmergefiles(sname, informpath,trackinform, tmpfd,...
    difallfd,expectedTablesnum, MergeConfig, logstring)     
    %
    MergeTblsDate = [];
    MergeTbls = [];
    dRfd = [];
    %
    % if Results folder does not exist but there are enough inform files to
    % generate output then run merge function
    %
    Rfd = [informpath,'\Phenotyped\Results\Tables'];
    mergeroot = informpath;
    if ~exist(Rfd,'dir') && trackinform == length(tmpfd)
        MaSS(mergeroot, sname, MergeConfig, logstring);
    end
    lf = fullfile(Rfd,'MaSSLog.txt');
    if exist(lf, 'file')
       fid = fopen(lf);
       if fid >= 0
           erl = textscan(fid,'%s','HeaderLines',...
               2,'EndofLine','\r', 'whitespace','\t\n');
           fclose(fid);
           %
           erl = erl{1};
           ii = contains(erl,'Error');
           if ~isempty(find(ii,1))
               delete([Rfd,'\*'])
           end
       end
    end
    %
    % check if folder exists
    %
    if exist(Rfd,'dir')
        % 
        Tablespath = [informpath,'\Phenotyped\Results\Tables\*_table.csv'];
        Tblsdate = dir(Tablespath);
        %
        Tablesnum = length(Tblsdate);
        %
        if ~isempty(Tblsdate)
            [~,idx] = max([Tblsdate.datenum]);
            Tblsdate = Tblsdate(idx);
            %
            dRfd = Tblsdate.date;
        end
        %
        % if results folder was created after most recent inform folder
        % rerun merge functions and create new results
        %
        if isempty(Tblsdate) || (datetime(dRfd) < datetime(difallfd)) ...
                || Tablesnum ~= expectedTablesnum
            MaSS(mergeroot, sname, MergeConfig, logstring);
            Tblsdate = dir(Tablespath);
            %
            if ~isempty(Tblsdate)
                [~,idx] = max([Tblsdate.datenum]);
                Tblsdate = Tblsdate(idx);
                %
                dRfd = Tblsdate.date;
            end
        end
        %
        % set up tracking functions
        %
        if ~isempty(Tblsdate)
            MergeTblsDate = Tblsdate.date(1:11);
            %
            Tablespath = [informpath,'\Phenotyped\Results\Tables\*_table.csv'];
            Tables = dir(Tablespath);
            Tablesnum = length(Tables);
            %
            MergeTbls = [num2str(Tablesnum),'of',num2str(expectedTablesnum)];
        end
    end
end