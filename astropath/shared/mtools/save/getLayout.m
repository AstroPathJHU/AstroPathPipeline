function [R,G] = getXMLLayout(C)
%%--------------------------------------------------------------
%% get a list of fields two ways, one from the annotations.xml,
%% the other is from the directory, and compare the two.
%%
%% 2019-01-07   Alex Szalay, Baltimore
%% 2020-06-08   Alex Szalay, fixed mismatch in plan and files
%%                  added matchPlanToFiles function, dropped fixM2
%%--------------------------------------------------------------
    %
    logMsg(C,mfilename);
    %
    % get the XML annotations with the layout plan
    %
    [r,G,p] = getXMLPlan(C);
    %
    % get the directory info of the image files with timestamps
    %
    t = getTimestamps(C);
    %   
    q = matchPlanToFiles(C,r,t);
    %    
    if (numel(q.cx)>numel(t.cx))
        msg = sprintf('WARNING: mismatch of plan(%d) and files(%d)',...
            numel(q.cx), numel(t.cx));
        logMsg(C,msg,1);        
        q = removeDuplicates(C,q);
    end
    %
    % join the tables, make sure that the timestamps also match
    % and select the distinct columns
    %
    R = join(t,q);
    dt = max(abs(uint64(R.t)-uint64(R.time)));
    if (dt>5)
        msg = sprintf('ERROR: Max time difference is %d seconds\n',dt,1);
        logMsg(C,msg);
    end
    %
    R = sortrows(R,'t');
    R = R(:,[4:8,1:3,10]);
    %
    % add sequential numbering
    %
    R.n = (1:numel(R.cx))';
    %
end

