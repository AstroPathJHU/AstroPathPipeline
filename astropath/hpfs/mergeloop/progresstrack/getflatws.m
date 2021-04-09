%% getflatws
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 02/25/2019
%% --------------------------------------------------------------
%% Description:
%%% for a give specimen in a directory; check if the status of the
%%% flatwarping of the im3 images
%% --------------------------------------------------------------
%%
function[flatwdate,actualflatwnum, expectflatwnum, alg] = ...
    getflatws(wd,sname, actualim3num, main,tmpfd)
        flatwdate = [];
        actualflatwnum = 0;
        expectflatwnum = 0;
        alg = repmat({''},1,numel(tmpfd));
        %
        % flatw im3 location path
        %
        flatwpath = [wd,'\',sname,'\im3\flatw'];
        %
        % find the number of im3 files created and their date
        %
        if exist(flatwpath,'dir')
            flatw = dir([flatwpath,'\*.im3']);
            %
            % check if flatw ran for all images 
            %
            if length(flatw) ~= actualim3num
                %
                % if number of *.im3's do not equal number of flatw *.im3's
                % then there was an error in flat code or it is not finished
                % delete flatw tracking and move on 
                %
                flatw = [];
                flatwdate = 'NA';
                %}
            else
                %
                % if flatw code ran correctly use most recent date
                %
                [~,idx] = max([flatw(:).datenum]);
                flatwdate = flatw(idx).date(1:11);
                %
                % populate that Specimen to the inForm_queue if its not
                % already there
                %
                alg = pop_queue(wd, sname, main, tmpfd);
            end
            actualflatwnum = length(flatw);
            expectflatwnum = actualim3num;
        end
end