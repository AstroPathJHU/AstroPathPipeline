function P = getXMLPolygonAnnotations2(C)
%%function P = getXMLPolygonAnnotations2(C, varargin)
%%-----------------------------------------------------------
%% get the .annotations.polygons.xml file, if it is present
%% Pack the results inot three tables:
%%    P.A: Annotations
%%    P.R: Regions
%%    P.V: Vertices
%%
%% Alex Szalay, Baltimore, 2019-02-12
%%-----------------------------------------------------------
    %{
    logMsg(C.samp,mfilename);
    %
    flag = 0;
    if (numel(varargin)==1)
        flag=1;
    end
    %
    xpath = [C.root,'\',C.samp,'\im3\',C.scan,...
        '\',C.samp,'_',C.scan ,'.annotations.polygons.xml'];
    %}
    %
    C.samp = 'M36_1';
    xpath = 'M36_1_Scan1.detailed.annotations.polygons.xml';
    %
    if (exist(xpath)==0)
        fprintf(1,'%s annotations file not found\n',C.samp);
        return    
    end
    %
    try 
        s = xml2struct(xpath);
    catch
        error('Could not parse XML file %s\n',xpath);
        return
    end
    %
    a = s.Annotations.Annotation;
    %
    % loop through the annotations arrays
    %
    A = [];
    R = [];
    V = [];
    for n = 1:numel(a)
        %
        at  = getAttributes(a,n,0);
        A   = [A;at];
        %
        rt  = getRegions(a,n,0);
        R   = [R;rt];
        %
        vt  = getVertices(a,n,0);
        V   = [V;vt];
        %
    end
    P.a = a;
    P.A = A;
    P.R = R;
    P.V = V;
    %
end


function q = getAttributes(a,n,samp)
%%--------------------------------------
%% get the Attributes for a given Layer
%%
%% Alex Szalay, Baltimore, 2019-02-19
%%--------------------------------------
    %
    p = a{n}.Attributes;
    r = a{n}.Regions;
    cc = dec2hex(str2num(p.LineColor),6);
    cc = string([cc(5:6),cc(3:4),cc(1:2)]);
    %
    nreg = numel(r.Region);
    q = table(samp,n,string(p.Name),cc,strcmp(p.Visible,'True'),nreg);
    q.Properties.VariableNames = {'sampleid','layer','name',...
        'color','visible','nreg'};
    %    
end


function r = getRegions(a,n,samp)
%%-----------------------------------------
%% get the Regions, pack them into a table
%%
%% Alex Szalay, Baltimore, 2019-02-12
%%-----------------------------------------
    %
    b = a{n}.Regions.Region;
    r    = [];
    for m=1:numel(b)
        p = b{m}.Attributes;
        rid = uint32(m+1000*n+10000*samp);
        nv = uint32(numel(b{m}.Vertices.V));
        bin = string("0000000");
        ra = table(rid,samp,n,m,str2num(p.NegativeROA),...
            string(p.Type),nv,bin);
        ra.Properties.VariableNames = {'regionid','sampleid','layer',...
            'rid','isNeg','type','nvert','bin'};
        r  = [r;ra];        
    end
    %
end


function v = getVertices(a,n,samp)
%%---------------------------------------------
%% get all the vertices from a given Region set
%% Positive are counterclockwise
%% Holes are counterclockwise
%%
%% Alex Szalay, Baltimore, 2019-02-12
%%---------------------------------------------
    %
    % get the Region array
    %
    b = a{n}.Regions.Region;
    v = [];
    for m=1:numel(b)
        %
        % get the Vertices
        %
        p = b{m}.Vertices.V;
        rid = [];
        vid = [];
        x   = [];
        y   = [];
        for k=1:numel(p)
            rid(k) = uint32(m+1000*n+10000*samp);
            vid(k) = uint32(k);
            x(k)   = uint32(str2num(p{k}.Attributes.X));
            y(k)   = uint32(str2num(p{k}.Attributes.Y));
        end
        %
        %fprintf(1,'%d %d %d %d\n',n,m,numel(p),size(x,2));
        %
        va = table(rid',vid',x',y');
        va.Properties.VariableNames = {'regionid','vid','x','y'};
        v  = [v;va];
        %
    end
    %
end

