function [r] = getXMLPlan(C)
%%----------------------------------------------------
%% read XML file with grid annotations
%% Daphne Schlesinger, 2017
%% edited by Alex Szalay, 2018-0725
%%----------------------------------------------------
%% take the path for an xmlfile corresponding to a qptiff. If the path
%% is not valid, a window opens to select a file. 
%% The filename, X position, Y position,  W width, and H height are written
%% into a table, the output of a function. The fields are:
%%       n,x,y,w,h,cx,cy,time,file
%% Eveything is in units of microns
%% The fields are sorted in the order they were observed
%%------------------------------------------------------
    %
    logMsg(C.samp,mfilename);
    %
    xpath = [C.root,'\',C.samp,'\im3\',C.scan,'\',C.samp,...
        '_', C.scan '_annotations.xml'];
    %
    % try/catch for invalid filepath
    %
    try
        s = xml2struct(xpath);
    catch
        error(sprintf('XML file %s not found',xpath));
    end
    %
    % get annotations array
    %
    A = s.AnnotationList.Annotations.Annotations_dash_i;
    n = 0;
    for i=1:numel(A)
        %
        fn = A{i}.History.History_dash_i{3}.Im3Path.Text;
        if (isempty(fn))
            continue;
        end
        n = n+1;
        %
        N(n) = n;
        X(n) = str2num(A{i}.Bounds.Origin.X.Text);
        Y(n) = str2num(A{i}.Bounds.Origin.Y.Text);
        W(n) = str2num(A{i}.Bounds.Size.Width.Text);
        H(n) = str2num(A{i}.Bounds.Size.Height.Text);
        %
        tt   = A{i}.History.History_dash_i{3}.TimeStamp.Text;
        if (strcmp(tt,'-05:00')>-1)
            tt = replace(tt,'-05:00','');
        end
        if (strcmp(tt,'-04:00')>-1)
            tt = replace(tt,'-04:00','');
        end
        T(n) = uint64(posixtime(datetime(tt,...
            'InputFormat','yyyy-MM-dd''T''HH:mm:ss.SSSSSS')))-1;
        F{n} = fn(5:end);
        %
    end
    %
    CX = round(X+0.5*W);
    CY = round(Y+0.5*H);
    %
    % create table from the results
    %
    r = table(N',X',Y',W',H',CX',CY',T',F');
    r.Properties.VariableNames = {'n','x','y',...
        'w','h','cx','cy','time','file'};
    %
end



