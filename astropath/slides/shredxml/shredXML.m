function C = shredXML(root,samp,varargin)
%%-------------------------------------------------------------------
%% Extract the XML information from the im3 files 
%% in a directory.  Can generate three different XML
%% files:
%%  <sample>.Full.xml - the whole XML of the first im3 file
%%  <sample>.Parameters.xml - the main parameters of the first file
%%  <sample>_[cx,cy].SpectralBasisInfo.Exposure.xml - exposure times
%%      for each HPF
%%  opt:
%%      0: generates all three
%%      1: skips the Full
%%
%% 2020-06-17   Alex Szalay
%%-------------------------------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %---------------------------
    % configure the environment
    %---------------------------
    C = getConfig(root,samp,'shredxml');
    logMsg(C,'shredxml started',1);
    %
    if (C.err>0)
        return
    end
    %-------------------------------
    % create the different paths
    %-------------------------------
    C.im3 = fullfile(C.root,C.samp,'im3');
    C.scan = getScan(C.root,C.samp);
    C.im3path = fullfile(C.im3,C.scan,'MSI');
    C.xmlpath = fullfile(C.im3,'xml');
    %
    if (exist(C.xmlpath)~=7)
        mkdir(C.xmlpath);
    end
    %
    %--------------------------------------------
    % configure the different extract options
    %--------------------------------------------
    pcode= '\\bki02\c\BKI\IM3Tools\ConvertIm3.exe ';
    px   = ' XML -x ';
    pfull= ' XML -t 64 ';
    pexp = './/G[@name=''SpectralBasisInfo'']//D[@name=''Exposure'']';
    ppmt = ['"//D[@name=''Shape'']  ',...
            '| //D[@name=''SampleLocation''] ',...
            '| //D[@name=''MillimetersPerPixel''] ',...
            '| (.//G[@name=''Protocol'']//G[@name=''CameraState''])[1]" '];
    pout = [' -o ',C.xmlpath];
    plog = [' >>','"',C.xmlpath,'\doShred.log"'];
    %
    fp = [C.im3path,'\*.im3'];
    d = dir(fp);
    if (isempty(d))
        msg = sprintf('ERROR: Empty im3 filepath %s',fp);
        logMsg(C,msg,1);
        return
    end    
    %   
    %-----------------------------------------
    % take the first file from the directory
    %-----------------------------------------
    fname = fullfile(d(1).folder,d(1).name);
    %--------------------
    % convert .Full.xml
    %--------------------
    cmd = [pcode, fname, pfull, pout, plog];
    runSysCmd(C,cmd);
    %rename the file
    src = fullfile(C.xmlpath,replace(d(1).name,'.im3','.xml'));
    dst = [C.xmlpath,'\',C.samp,'.Full.xml'];
    movefile(src,dst);
    %----------------------------------------------
    % optionally loop through all the im3 files
    % (enable this manually by changing the if)
    %----------------------------------------------
    if (1==0)
        for i=1:numel(d)
            fname = fullfile(d(i).folder,d(i).name);
            cmd = [pcode, fname, pfull, pout, plog];
            runSysCmd(C,cmd);
        end
    end
    %--------------------------
    % convert .Parameters.xml
    %--------------------------
    cmd = [pcode, fname, px, ppmt, pout, plog];
    runSysCmd(C,cmd);
    % rename the file
    src = fullfile(C.xmlpath,...
        replace(d(1).name,'.im3',...
        ['.Shape.SampleLocation.MillimetersPerPixel',...
        '.Protocol.CameraState.xml']));
    dst = [C.xmlpath,'\',C.samp,'.Parameters.xml']; 
    if (exist(src)>0)
        movefile(src,dst)
    end
    %------------------------
    % convert .Exposures.xml
    %------------------------
    for i=1:numel(d)
        %
        fname = fullfile(d(i).folder,d(i).name);
        cmd = [pcode, fname, px, pexp, pout, plog];
        runSysCmd(C,cmd);
        %
    end
    %
    d = dir(fullfile(C.xmlpath,'*.xml'));
    msg = sprintf('shredxml finished, created %d xml files ',numel(d));
    logMsg(C,msg,1);
    %
end


