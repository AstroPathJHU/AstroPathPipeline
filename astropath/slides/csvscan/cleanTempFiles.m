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

