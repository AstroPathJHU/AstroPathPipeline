
function C = scanPath(C,src,dst,tmp)
%%----------------------------------------------
%% execute the parser command on <src>, and
%% create output file at <dst>
%%
%% 2020-07-18   Alex Szalay
%%---------------------------------------------
    %
    cmd = [C.pcode,' ',src,' ',tmp,' ',C.samp];
    %fprintf('%s\n',cmd);
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
    %-------------------------------------
    % check if output ok, if yes, rename
    %-------------------------------------
    if (exist(tmp)>0)
        movefile(tmp,dst);
    else
        msg = sprintf('WARNING: No output from scanning %s failed',src);
        logMsg(C,msg,1);        
        return
    end
    %
end

