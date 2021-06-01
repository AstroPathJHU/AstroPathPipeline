function [R,G,P] = getXMLPlan(C)
%%----------------------------------------------------
%% read XML file with grid annotations
%%
%%----------------------------------------------------
%% take the path for an xmlfile corresponding to a qptiff. If the path
%% is not valid, a window opens to select a file. 
%% The filename, X position, Y position,  W width, and H height are written
%% into a table, the output of a function. The fields are:
%%       n,x,y,w,h,cx,cy,time,file
%% Eveything is read in units of microns, converted to pixels
%% The fields are sorted in the order they were observed
%% R is the table with the rectangles
%% G are global parameters
%% P is a table with the perimeters
%% 2017 Daphne Schlesinger
%% 2018-07-25   Alex Szalay
%% 2020-04-15   Alex Szalay, modified to accommodate new Akoya XML format
%%------------------------------------------------------------------------
    %
    logMsg(C,mfilename);
    %
    xp = [C.root,'\',C.samp,'\im3\',C.scan,'\',C.samp,...
        '_', C.scan '_annotations.xml'];
    %
    % try/catch for invalid filepath
    %
    try
        s = xml2struct(xp);
    catch
        msg = sprintf('XML file %s not found',xp));
        logMsg(C,msg);
        C.err=1;
        return
    end
    %
    % get annotations_i as a cell array
    %
    a = s.AnnotationList.Annotations.Annotations_dash_i;
    if (strcmp(class(a),'struct')==1)
        a = {a};
    end
    %
    % test for the version number
    %
    new = false;
    for i=1:numel(a)
        if (strcmp(a{i}.Attributes.subtype,'ROIAnnotation')==1)
            new = true;
        end
    end
    %---------------------------------
    % loop through the region array
    %---------------------------------
    G = [];
    P = [];
    R = [];
    if (new)
        %
        na = numel(a);
        kn = 0;
        for n = 1:na
            %
            % ignore RectangleAnnotation
            %
            %fprintf('%d, %s\n', n, a{n}.Attributes.subtype);
            if (strcmp(a{n}.Attributes.subtype,'RectangleAnnotation')==1)
                continue;
            end
            %
            b = a{n};    
            %---------------------------
            % loop through the Fields
            %---------------------------
            if (strcmp(class(R),'table')==1)
                kR = height(R);
            else
                kR = 0;
            end
            %
            ff = b.Fields.Fields_dash_i;
            if (strcmp(class(ff),'struct')==1)
                ff = {ff};
            end
            %
            r = getFields(ff,kR);
            R = [R;r];
            if (numel(r)==0)
                continue
            end
            kn = kn+1;
            %-------------------------------------
            % read the globals and the perimeter
            %-------------------------------------
            G = [G;getGlobals(b,kn)];
            %
            p = b.Perimeter.Perimeter_dash_i;
            P = [P;getPerimeter(p,kn)];
            %
        end
    else
        R = getFields(a,0);
    end
end


function R = getFields(Fields,n)
%%---------------------------------------
%% get the info from the Fields array
%%---------------------------------------
    %
    for i=1:numel(Fields)
        %
        f = Fields{i};
        %
        % get the filename
        %
        nh = numel(f.History.History_dash_i);        
        h  = f.History.History_dash_i{nh};
        %
        if (strcmp(h.Type.Text,'Acquired')==0)
            continue
        end
        fn = h.Im3Path.Text;
        %
        % get the timestamp
        %
        tt   = h.TimeStamp.Text;
        if (strcmp(tt,'-05:00')>-1)
            tt = replace(tt,'-05:00','');
        end
        if (strcmp(tt,'-04:00')>-1)
            tt = replace(tt,'-04:00','');
        end
        tt = (posixtime(datetime(tt,...
            'InputFormat','yyyy-MM-dd''T''HH:mm:ss.SSSSSS')))-1;
        %
        if (isempty(fn))
            continue;
        end
        n = n+1;
        %
        % get the coordinates
        %
        N(n) = n;
        X(n) = str2num(f.Bounds.Origin.X.Text);
        Y(n) = str2num(f.Bounds.Origin.Y.Text);
        W(n) = str2num(f.Bounds.Size.Width.Text);
        H(n) = str2num(f.Bounds.Size.Height.Text);
        %
        T(n) = tt;
        F{n} = fn(5:end);
        %
    end
    %
    if (n==0)
        R =[];
        return
    end
    %
    CX = round(X+0.5*W);
    CY = round(Y+0.5*H);
    %
    R = table(N',X',Y',W',H',CX',CY',T',F');
    R.Properties.VariableNames = {'n','x','y',...
        'w','h','cx','cy','time','file'};
    %
end




function G=getGlobals(a,n)
%%-------------------------------
%% get the global parameters
%%-------------------------------
    %
    N  = n;
    X0 = str2num(a.Bounds.Origin.X.Text);
    Y0 = str2num(a.Bounds.Origin.Y.Text);
    Width  = str2num(a.Bounds.Size.Width.Text);
    Height = str2num(a.Bounds.Size.Height.Text);
    Unit   = 'microns';
    Tc     = a.History.History_dash_i.TimeStamp.Text;
    %
    G = cell2table({N,X0,Y0,Width,Height,Unit,Tc});
    G.Properties.VariableNames = {'M','X0','Y0',...
        'Width','Height','Unit','Tc'};    
    %
end


function P = getPerimeter(p,n)
%%--------------------------------
%% get the tissue perimeter
%%--------------------------------
    %
    for i=1:numel(p)
        rn(i) = n;
        pn(i) = i;
        px(i) =  str2num(p{i}.X.Text);
        py(i) =  str2num(p{i}.Y.Text);
    end
    %
    % create table from the results
    %
    P = table(rn',pn',px',py');
    P.Properties.VariableNames = {'m','n','x','y'};
    %
end