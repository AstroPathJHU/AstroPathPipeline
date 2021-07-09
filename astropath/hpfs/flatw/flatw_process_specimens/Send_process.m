function Send_process(main)
%
% Check if a new bki05 can be started
%
if exist([main,'\across_project_queues\process_flatw_queue.csv'], 'file')
    tbl = readtable([main, '\across_project_queues\process_flatw_queue.csv'],...
        'Delimiter' , ',',...
        'ReadVariableNames', true, 'format', repmat('%s ',1,16));
else
    return
end
%
% get the lines with empty 'processing sent' and 'process finished' columns
% but a filled flatfield.bin column
%
ii = strcmp(tbl.Processing_flatw_Sent,'') &...
    ~strcmp(tbl.flatfield_binfile_date,'') &...
    strcmp(tbl.Processing_flatw_Finished,'');
%
idx = find(ii);
tbl2 = tbl(ii,:);
if isempty(tbl2)
    return
end
%
drives = {'e$'};
%
% get only drives with flatw_go and process_flatw_queue in them
%
drives_flatw_go = strcat('\\bki09\',drives,...
    '\Processing_Specimens\flatw_go.txt');
drives_pqt = strcat('\\bki09\',drives,...
    '\Processing_Specimens\process_flatw_queue.csv');
ii = cellfun(@(x)exist(x, 'file'), drives_flatw_go) &...
        cellfun(@(x)exist(x, 'file'), drives_pqt);
drives = drives(ii);
%
i1 = 0;
%
% check each processing queue on bki05
%
for i2 = 1:length(drives)
    %
    d = ['\\bki09\',drives{i2},'\Processing_Specimens'];
    %
    % open the queue on that drive and collect only lines with a
    % specimen name filled in and lines without a finished tag
    %
    tbl3 = readtable([d, '\process_flatw_queue.csv'], 'Delimiter' , ',',...
        'ReadVariableNames', true, 'format', repmat('%s ',1,16));
    ii = ~strcmp(tbl3.Sample,'') & ...
        strcmp(tbl3.Processing_flatw_Finished,'');
    tbl4 = tbl3(ii,:);
    %
    if height(tbl4) < 2
        i1 = i1 + 1;
        %
        % populate to bki05
        %
        sp1 = tbl2(i1,:);
        idx1 = idx(i1);
        %
        sp1.Processing_flatw_Sent = {['Processing Sent to ',d]};
        t = datestr(datetime());
        t = t(1:11);
        sp1.Processing_flatw_Start = {t};
        tbl3 = [tbl3;sp1];
        %
        % try to write into the file 10 times once the file writes it
        % will exit
        %
        p = 0;
        tt = 1;
        while p == 0 && tt <10
            try
                writetable(tbl3,[d,'\process_flatw_queue.csv']);
                p = 1;
            catch
            end
            tt = tt + 1;
        end
        %
        if p == 0
            fprintf(['warning failed to write to ',d,'\n'])
            continue
        end
        %
        % if populated to bki05 then add processing and time to
        % bki04 processing queue
        %
        tbl.Processing_flatw_Sent(idx1) = {['Processing Sent to ',d]};
        tbl.Processing_flatw_Start(idx1) = {t};
        %
        % try to write into the file 10 times once the file writes it
        % will exit
        %
        p = 0;
        tt = 1;
        while p == 0 && tt <10
            try
                writetable(tbl,[main,'\across_project_queues\process_flatw_queue.csv']);
                p = 1;
            catch
            end
            tt = tt + 1;
        end
        %
        if p == 0
            fprintf('warning failed to write to main queue\n')
        end
    end
end
%
end