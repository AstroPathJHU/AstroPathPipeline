function C = scanCells(root,samp,varargin)
%%-------------------------------------------------------------------
%% Create the loadfiles.csv that drives the loading of the database.
%% Scans the different datasets containing csv files:
%%  dbload, geom, inform_data\Phenotyped\Results/Tables, 
%% opt: optional argument to replace 'Tables' in the inform results 
%%      subdirectory, to provide alternative processing options.
%%
%% 2020-07-18   Alex Szalay
%%-------------------------------
    %-------------------------------
    % create the different paths
    %-------------------------------------------------------------------
    %
    opt = [];
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %---------------------------
    % configure the environment
    %---------------------------
    C = getConfig(root,samp,'csvscan');
    logMsg(C,'scanCells started',1);
    if (C.err>0)
        return
    end
    if (isempty(opt))
        C.cellpath = fullfile(C.root,C.samp,...
            'inform_data','Phenotyped','Results','Tables');
    else
        C.cellpath = fullfile(C.root,C.samp,...
            'inform_data','Phenotyped','Results',opt);
    end
    C.geompath = fullfile(C.root,C.samp,'geom');
    %
    C.loadfile = fullfile(C.dbload,[C.samp,'_loadfiles.csv']);
    C.tmppath  = fullfile('\\bki02','f','tmp');
    C.tmpfile  = fullfile(C.tmppath,[C.samp,'_loadfiles.csv']);
    C.tmp1 = replace(C.tmpfile,'loadfiles','1');
    C.tmp2 = replace(C.tmpfile,'loadfiles','2');
    C.tmp3 = replace(C.tmpfile,'loadfiles','3');    
    %-------------------------------------------------
    % check whether the tmppath exists, if not, create
    %-------------------------------------------------
    if (exist(C.tmppath)~=7)
        mkdir(C.tmppath);
    end
    %-----------------------------------
    % make sure that the dbload exists
    %-----------------------------------
    if (exist(C.dbload)~=7)
        msg = sprintf('ERROR: %s subdirectory not found',C.dbload);
        logMsg(C,msg,1);
        return
    end
    %
    if (exist(C.loadfile)>0)
        msg = sprintf('Old %s deleted',C.loadfile);
        logMsg(C,msg);
        delete(C.loadfile);
    end
    %---------------------------------------------------------
    % clean up previous versions of the loadfiles, if any
    %---------------------------------------------------------
    cleanTempFiles(C,1);
    %--------------------------------------------------
    % create the command for the parsing
    %--------------------------------------------------
    C.pcode= '\\bki02\c\BKI\bin\cellparser.exe ';
    %-------------------------------------
    % execute it on the source directories
    %-------------------------------------
    src = {C.dbload, C.cellpath, C.geompath};
    dst = {C.tmp1, C.tmp2, C.tmp3};
    for i=1:numel(src)
        C = scanPath(C,src{i},dst{i},C.tmpfile);
        if (C.err==1)
            return
        end
    end
    %------------------------------------
    % merge the files together
    %------------------------------------
    cmd = '@';
    for i=1:numel(dst)
        cmd = [cmd,' + ',dst{i}];
    end
    cmd = replace(cmd,'@ +','COPY');
    cmd = [cmd,' /a ', C.loadfile, ' /b /y >>out.txt'];
    %fprintf('%s\n',cmd);
    runSysCmd(C,cmd);
    %------------------------------------------------------
    % clean up previous versions of the loadfiles, if any
    %------------------------------------------------------
    cleanTempFiles(C,0);
    %
    msg = sprintf('scanSample finished',1);
    logMsg(C,msg);
    %
end


%{
function cleanTempFiles(C, flag)
%%--------------------------------------------------
%% Clean up the previous version of the loadfiles.
%% Write message only if flag==1
%%
%% 2020-07-18   Alex Szalay
%%--------------------------------------------------
    %
    tempfiles = {C.tmpfile, C.tmp1, C.tmp2, C.tmp3};
    for i=1:numel(tempfiles)
        temp = tempfiles{i};
        if (exist(temp)>0)
            delete(temp);
            if (flag==1)
                msg = sprintf('Old %s deleted',temp);
                logMsg(C,msg);
            end
        end
    end
    %
    if (exist('out.tmp')>0)
        delete ('out.tmp');
    end
    %
end
%}