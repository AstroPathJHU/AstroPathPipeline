function p = getXMLParams(C)
%%----------------------------------------------
%% read the global sample.Parameters.xml file
%% and extract the params
%% Alex Szalay, Baltimore, 2020-04-14
%%----------------------------------------------
    logMsg(C.samp,mfilename);
    %
    fn = [C.root '\' C.samp '\im3\xml\' C.samp '.Parameters.xml'];
    %
    try
        fp  = fopen(fn);
        xml = textscan(fp,'%s','Delimiter','\n');
        fclose(fp);
    catch
        s = sprintf('ERROR: XML file %s not found\n', fn);
        logMsg(C.samp,s);
        p.err=-1;
        return
    end
    %
    p.err = 0;
    x = xml{1};
    for i=1:numel(x)
        s = x{i};
        if (numel(regexp(s,'"Shape"'))>0)
            %
            q = regexp(s,'>.*<','match');
            q = replace(replace(q{1},'<',''),'>','');
            p.shape = str2num(q);
            %
        end
        %
        if (numel(regexp(s,'"SampleLocation"'))>0)
            %
            q = regexp(s,'>.*<','match');
            q = replace(replace(q{1},'<',''),'>','');
            p.location = str2num(q);
            %
        end
        %
        if (numel(regexp(s,'"MillimetersPerPixel"'))>0)
            %
            q = regexp(s,'>.*<','match');
            q = replace(replace(q{1},'<',''),'>','');
            p.pscale = 1.0/str2num(q)/1000;
            %
        end
        %        
    end
    %
end
