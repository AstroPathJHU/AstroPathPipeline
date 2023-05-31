function status = runSysCmd(C,cmd)
%%----------------------
%% run an OS command
%%----------------------
    %
    %fprintf('%s\n',cmd);
    [status, cmdout] = system(cmd,'-echo');
    %
    if (status>0)
        logMsg(C,'ERROR while executing runSysCmd',1);
    end
    %
end