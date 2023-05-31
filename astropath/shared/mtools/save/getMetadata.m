function C = getMetadata(C)
%%--------------------------------------------------
%% read the various annotations and QPTIFF files,
%% extract the information and save it to csv.
%%
%% Alex Szalay, Baltimore, 2018-10-29
%%--------------------------------------------------
    %
    logMsg(C,mfilename);
    %
    % make sure that layout filenames are compatible with sample
    %
    s = C.R.file{1};
    s = s(1:strfind(s,'_[')-1);
    if (strcmp(s,C.samp)==0)
        msg = sprintf('ERROR: Sample in annotations.xml is %s\n',s);
        logMsg(C,msg);
        C.err = 1;
    end
    if (C.opt==-4)
        return
    end
    %
end
