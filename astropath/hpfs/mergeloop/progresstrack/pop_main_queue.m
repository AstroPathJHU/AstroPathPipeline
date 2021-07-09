%% pop_main,queue
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% populate the main inForm_queue if the files have not yet been processed,
%%% queue location is in the 'main' folder
%% --------------------------------------------------------------------
%%
function[] = pop_main_queue(wd, main)
    %
    % if main_inform_queue does not exist create it
    %
    str = 'Path,Specimen,Antibody,Algorithm,Processing \r\n';
    %
    if ~exist([main,'\across_project_queues\inform-queue.csv'], 'file')
        f = fopen([main,'\across_project_queues\inform-queue.csv'], 'w' );  
           fprintf(f,str);
        fclose(f);
    end
    %
    % if clinical specimen inForm queue does not exist create it
    %
    if ~exist([wd,'\upkeep_and_progress\inForm_queue.csv'], 'file')
            f = fopen([wd,'\upkeep_and_progress\inForm_queue.csv'], 'w' );
            fprintf(f,str);
            fclose(f);
    end
    %
    str = 'Path,Specimen,Antibody,Algorithm,Processing ';
    %
    % open the main inForm queue
    %
    fileID = fopen([main,'\across_project_queues\inform-queue.csv']);
    queuef = textscan(fileID,'%s','HeaderLines',...
        1,'EndofLine','\r', 'whitespace','\t\n');
    fclose(fileID);
    queuef_main = queuef{1};
    %
    % trim all the white space and if the string contains two 'Processing'
    % strings assume the second string ran properly 
    %
    del_rows = [];
    %
    for i1 = 1:length(queuef_main)
        x = queuef_main{i1};
        x = strsplit(x,',');
        if isempty(x{1}) || length(x{1}) > 2000
            del_rows = [del_rows,i1];
            continue
        end
        x = cellfun(@(x)strtrim(x),x,'Uni',0);
        x = strjoin(x,', ');
        if count(x,'Processing') > 1
            str1 = extractBetween(x,'Processing','Processing');
            x = replace(x,strcat('Processing',str1),'');
        end
        queuef_main{i1} = x;
    end
    %
    if ~isempty(del_rows)
        queuef_main(del_rows) = [];
    end
    %
    % open the clinical specimen inForm queue
    %
    fileID = fopen([wd,'\upkeep_and_progress\inForm_queue.csv']);
    queuef = textscan(fileID,'%s','HeaderLines',...
        1,'EndofLine','\r', 'whitespace','\t\n');
    fclose(fileID);
    queuef_CS = queuef{1};
    %
    % trim all the white space and if the string contains two 'Processing'
    % strings assume the second string ran properly 
    %
    del_rows = [];
    queuef_CS_check = cell(length(queuef_CS),1);
    %
    for i1 = 1:length(queuef_CS)
        x = queuef_CS{i1};
        x = strsplit(x,',');
         if isempty(x{1}) || length(x{1}) > 2000
            del_rows = [del_rows,i1];
            continue
        end
        x = cellfun(@(x)strtrim(x),x,'Uni',0);
        queuef_CS_check{i1} = strjoin(x(1:3),', ');
        x = strjoin(x,', ');
        if count(x,'Processing') > 1
            str1 = extractBetween(x,'Processing','Processing');
            x = replace(x,strcat('Processing',str1),'');
        end
        queuef_CS{i1} = x;
    end
    %
    % remove blank rows
    %
    if ~isempty(del_rows)
        queuef_CS(del_rows) = [];
        queuef_CS_check(del_rows) = [];
    end
    %
    % get the unique lines from CS queue
    %
    D = unique(queuef_CS_check(:));
    %
    % update the inform queues for lines that have been added more than once 
    %
    for i1 = 1:length(D)
        %
        % check for each unique line in both queues
        %
        current_line = D(i1);
        ii = strcmp(current_line, queuef_CS_check);
        CS_line_numbers = find(ii);
        CS_line_counts = numel(CS_line_numbers);
        %
        ii = contains(queuef_main,[current_line{1},',']);
        M_line_numbers = find(ii);
        M_line_counts = numel(M_line_numbers);
        %
        % for each line in CS queue check if the number of lines is less
        % than the number in main queue. If it is then assume the rows are
        % in matching order. Check if the current line in CSn or in main 
        % queue is longer; if it is longer in the CS queue replace
        % in the main queue otherwise use what is in the main queue for
        % both. If the number of matching lines found in the CS queue 
        % exceeds the length of lines found in the main queue add the end 
        % of the main queue.
        %
        for i2 = 1:CS_line_counts
            CSn = queuef_CS(CS_line_numbers(i2));
            len_CSn = length(CSn{:});
            %
            if i2 <= M_line_counts
                Mn = queuef_main(M_line_numbers(i2));
                len_Mn = length(Mn{:});
                %
                if len_CSn > len_Mn
                    queuef_main(M_line_numbers(i2)) = CSn;
                else
                    queuef_CS(CS_line_numbers(i2)) = Mn;
                end
            else
                queuef_main(end+1) = CSn;
            end
        end
        %
        if M_line_counts > CS_line_counts
            for i2 = CS_line_counts + 1:M_line_counts
                Mn = queuef_main(M_line_numbers(i2));
                queuef_CS(end+1) = Mn;
            end
        end
    end
    %
    [nrow, ncol] = size(queuef_main);
    if ncol > nrow
        queuef_main = queuef_main';
    end
    queuef_main = replace(queuef_main,'\','\\');
    queuef_main = [{str};queuef_main];
    %
    queuef_CS = replace(queuef_CS,'\','\\');
    queuef_CS_out = [{str};queuef_CS];
    %
    % remove the old file and write a new one
    %
    try
        %
        f = fopen([wd,'\upkeep_and_progress\inForm_queue.csv'], 'w' );
        cellfun(@(x) fprintf(f,[x,' \r\n']),queuef_CS_out);
        fclose(f);
        %
    catch
    end
    % 
    try
        %
        qf = [wd,'\upkeep_and_progress\Main_inForm_queue.csv'];
        mf = [main,'\across_project_queues\inform-queue.csv'];
        f = fopen(mf, 'w' );
        cellfun(@(x) fprintf(f,[x,' \r\n']),queuef_main);
        fclose(f);
        %
        copyfile(mf,qf);
    catch
    end
end