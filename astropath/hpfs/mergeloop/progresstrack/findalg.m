%% findlag
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% find the most recent algorithm used to analyze an antibody in inform
%% --------------------------------------------------------------------
%%
function alg = findalg(wd,sname,tmpfd)
    %
    % for each antibody find the algorithm related to the most recent
    % encoded line
    %
    alg = repmat({''},1,numel(tmpfd));
    fileID = fopen([wd,'\upkeep_and_progress\inForm_queue.csv']);
    queuef = textscan(fileID,'%s','HeaderLines',...
        2,'EndofLine','\r', 'whitespace','\t\n');
    fclose(fileID);
    queuef = queuef{1};
    %
    for i2 = 1:length(tmpfd)
        AB = tmpfd(i2).name;
        str = [wd,', ',sname,', ',AB,','];
        line1 = queuef(contains(queuef,str));
        %
        if ~isempty(line1)
            line1 = line1{end};
            line1 = extractBetween(line1,[AB,','],',');
            alg(i2) = line1;
        end
    end
end