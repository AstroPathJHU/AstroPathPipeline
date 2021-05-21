function C=scanCsv(C)
%%-------------------------------------------------------------
%% Will Scan the dbload directoruy for the calibration data
%% and prepare the project<prno>_loadfiles.csv
%%
%% 2020-08-08   Alex Szalay
%%-------------------------------------------------------------
    %
    C.loadfile = fullfile(C.dbload,sprintf('project%d_loadfiles.csv',...
        C.project));
    C.tmppath  = fullfile('\\bki02','f','tmp');
    C.tmpfile  = fullfile(C.tmppath,'Ctrl_loadfiles.csv');
    %-------------------------------------------------
    % check whether the tmppath exists, if not, create
    %-------------------------------------------------
    if (exist(C.tmppath)~=7)
        mkdir(C.tmppath);
    end
    %
    if (exist(C.loadfile)>0)
        msg = sprintf('Old %s deleted',C.loadfile);
        logMsg(C,msg);
        delete(C.loadfile);
    end    
    %
    %--------------------------------------------------
    % create and execute the command for the parsing
    %--------------------------------------------------
    C.pcode= '\\bki02\c\BKI\bin\cellparser.exe ';
    C = scanPath(C,C.dbload,C.tmppath);
    %
    if (exist(C.tmpfile))
        movefile(C.tmpfile,C.loadfile);
    else
        logMsg(C,['ERROR: cannot find ',C.tmpfile],1);
        return
    end
    %
end


function C = scanPath(C,src,tmp)
%%----------------------------------------------
%% execute the parser command on <src>, and
%% create output file at <dst>
%%
%% 2020-07-18   Alex Szalay
%%---------------------------------------------
    %
    samp = C.samp;
    if (isempty(samp))
        samp=sprintf('project%d',C.project);
        samp = 'Ctrl';
    end
    cmd = [C.pcode,' ',src,' ',tmp,' ',samp];
    fprintf('%s\n',cmd);
    status = runSysCmd(C,cmd);
    %-----------------------
    % test execution status
    %-----------------------
    if (status>0)
        msg = sprintf('ERROR: Scanning %s failed',src);
        logMsg(C,msg,1);
        C.err=1;
        return
    else
        msg = sprintf('Scanned %s files',src);
        logMsg(C,msg);
    end
    %
end

