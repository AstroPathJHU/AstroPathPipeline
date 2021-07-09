%% process_flt2bin
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 05/14/2019
%% --------------------------------------------------------------
%% Description:
%%% for a batch of specimens in a directory; check the status of the
%%% flatwarping .bin image. if it is not already created, make it
%% --------------------------------------------------------------
%%
function process_flt2bin(main)
%
tbl = readtable([main, '\AstropathPaths.csv'], 'Delimiter' , ',',...
    'ReadVariableNames', true);
%
for i1 = 1:height(tbl)
    %
    % Clinical_Specimen folder
    %
    wd = tbl(i1,:);
    wd = ['\\', wd.Dpath{1},'\', wd.Dname{1}];
    %
    samplenames = find_specimens(wd);
    %
    % get scan path and batchID of each sample
    %
    tbl2 = getSampleTable(wd, samplenames);
    %
    % get unique BatchIDs and cycle through to see if .flt file exists
    % for that batch. If it does not check for the mean flat field 
    % files of each specimen
    %
    Batches = unique(tbl2.BatchID);
    %
    if ~exist([wd,'\flatfield'],'dir')
        mkdir([wd,'\flatfield'])
    end
    %
    for i2 = 1:length(Batches)
        B1 = Batches{i2};
        %
        % check if the final .flt file exists for the batch
        %
        p1 = [wd,'\flatfield\flatfield_BatchID_',B1,'.bin'];
        if exist(p1,'file')
            continue
        end
        [ii2, tbl3, fnms] = checkMean(wd, B1, tbl2);
        %
        % number of slides in this batch
        %
        f = dir([wd,'\upkeep_and_progress\AstropathAPIDdef_*']);
        f = fullfile(f(1).folder,f(1).name);
        tbl4 = readtable(f);
        ii4 = tbl4.BatchID == str2double(B1);
        %
        % if all mean.flt files exist for every specimen then create a total
        % .bin flat field file
        %
        if sum(ii2)==0 && length(ii2) == sum(ii4)
            %
            fltOneBatch(tbl3, p1, fnms)
            %
        end
        %
    end
    %
end
%
end
%
