%% pop_queue
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 02/25/2019
%% --------------------------------------------------------------------
%% Description
%%% populate the inForm_queue if the files have not yet been processed,
%%% queue location is in the 'main' folder
%% --------------------------------------------------------------------
%%
function alg = pop_queue(wd, sname, main, tmpfd)
    %
    % first check if any inform output already exists
    %
    try
        alg = findalg(wd,sname, tmpfd);
    catch
        alg = repmat({''},1,numel(tmpfd));
    end
    ii = zeros(length(tmpfd),1);
    for i2 = 1:length(tmpfd)
        AB = tmpfd(i2).name;
        wd1 = [wd,'\',sname,'\inform_data\Phenotyped\',AB];
        ii(i2) = exist(wd1, 'dir');
    end
    ii2 = find(~ii);
    if sum(ii2) == 0
        return
    end
    %
    % if inform_queue does not exist create it
    %
    if ~exist([wd,'\upkeep_and_progress\inForm_queue.csv'], 'file')
        f = fopen([wd,'\upkeep_and_progress\inForm_queue.csv'], 'w' );  
         str = 'Path,Specimen,Antibody,Algorithm,Processing \r\n';
           fprintf(f,str);
        fclose(f);
    end
    %
    % open the inForm queue
    %
    fileID = fopen([wd,'\upkeep_and_progress\inForm_queue.csv']);
    queuef = textscan(fileID,'%s','HeaderLines',...
        1,'EndofLine','\r', 'whitespace','\t\n');
    fclose(fileID);
    queuef = queuef{1};
    queuef = cellfun(@(x)strrep(x,' ' ,''),queuef,'Uni',0);
    
    %
    % add those files to the queue if they are not already there
    %
    NewABs = tmpfd(ii2);
    %
    for i2 = 1:length(NewABs)
        if ~sum(contains(queuef,[sname,',',NewABs(i2).name,',']),1)
            i3 = replace(wd,'\','\\');
            fileID = fopen([wd,'\upkeep_and_progress\inForm_queue.csv'], 'a');
            AB = NewABs(i2).name;
            str = [i3,',',sname,',',AB,', , \r\n'];
            try
                fprintf(fileID,str);
                fclose(fileID);
            catch
            end
        end
    end
    alg = findalg(wd,sname,tmpfd);
end