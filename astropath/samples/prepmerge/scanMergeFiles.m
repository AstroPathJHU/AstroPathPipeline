function C=scanMergeFiles(C)
%%------------------------------------------------
%% scan the csv files and create a loadfiles.csv
%% for all the Batch files
%%
%% 2020-07-30   Alex Szalay
%%------------------------------------------------
    %
    logMsg(C,'scanMergeFiles started');
    %-----------------------------------
    % delete old loadfile, if it exists
    %-----------------------------------
    if (exist(C.loadfile)>0)
        msg = sprintf('Old %s deleted',C.loadfile);
        logMsg(C,msg);
        delete(C.loadfile);
    end
    %---------------------------------
    % build tmppath and tmpfile names
    %---------------------------------
    C.tmppath = fullfile('\\bki02','f','tmp');    
    C.tmpfile = fullfile(C.tmppath,[C.samp, '_loadfiles.csv']);
    C.tmp1 = fullfile(C.tmppath,[C.samp,'-1','_loadfiles.csv']);
    C.tmp2 = fullfile(C.tmppath,[C.samp,'-2','_loadfiles.csv']);
    C.tmp3 = fullfile(C.tmppath,[C.samp,'-3','_loadfiles.csv']);
    %-------------------------------------------------
    % check whether the tmppath exists, if not, create
    %-------------------------------------------------
    if (exist(C.tmppath)~=7)
        mkdir(C.tmppath);
    end
    %----------------------------------------------------------
    % create the command for the parsing the Batch directory
    %----------------------------------------------------------
    C.pcode= '\\bki02\c\BKI\bin\cellparser.exe '; 
    cmd = [C.pcode,' ',C.batch,' ',C.tmp1,' ',C.samp,'-1'];
    %fprintf('%s\n',cmd);
    status = runSysCmd(C,cmd);
    %-----------------------
    % test execution status
    %-----------------------
    if (status>0)
        msg = sprintf('ERROR: Scanning %s failed',C.batch);
        logMsg(C,msg,1);
        C.err=1;
        return
    else
        msg = sprintf('Scanned %s files',C.batch);
        logMsg(C,msg);
    end
    %----------------------------------------------------------
    % create the command for the parsing the Clinical directory
    %----------------------------------------------------------
    cmd = [C.pcode,' ',C.clinical,' ',C.tmp2,' ',C.samp,'-2'];
    %fprintf('%s\n',cmd);
    status = runSysCmd(C,cmd);
    %-----------------------
    % test execution status
    %-----------------------
    if (status>0)
        msg = sprintf('ERROR: Scanning %s failed',C.clinical);
        logMsg(C,msg,1);
        C.err=1;
        return
    else
        msg = sprintf('Scanned %s files',C.clinical);
        logMsg(C,msg);
    end
    %
    %----------------------------------------------------------
    % create the command for the parsing the Ctrl directory
    %----------------------------------------------------------
    cmd = [C.pcode,' ',C.ctrl,' ',C.tmp3,' ',C.samp,'-3'];
    %fprintf('%s\n',cmd);
    status = runSysCmd(C,cmd);
    %-----------------------
    % test execution status
    %-----------------------
    if (status>0)
        msg = sprintf('ERROR: Scanning %s failed',C.ctrl);
        logMsg(C,msg,1);
        C.err=1;
        return
    else
        msg = sprintf('Scanned %s files',C.ctrl);
        logMsg(C,msg);
    end
    %
    %------------------------------------
    % merge the files together
    %------------------------------------
    cmd = '@';
    dst = {C.tmp1, C.tmp2, C.tmp3};
    %dst = {C.tmp1, C.tmp2};
    for i=1:numel(dst)
        cmd = [cmd,' + ',dst{i}];
    end
    cmd = replace(cmd,'@ +','COPY');
    cmd = [cmd,' /a ', C.loadfile, ' /b /y >>out.txt'];
    %fprintf('%s\n',cmd);
    runSysCmd(C,cmd);
    %
    cleanTempFiles(C,0);
    %
    logMsg(C,'scanMergeFiles finished');
    %
end



function cleanTempFiles(C, flag)
%%--------------------------------------------------
%% Clean up the previous version of the loadfiles.
%% Write message only if flag==1
%%
%% 2020-07-18   Alex Szalay
%%--------------------------------------------------
    %
    tempfiles = {C.tmp1, C.tmp2, C.tmp3, 'out.txt'};
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
end
