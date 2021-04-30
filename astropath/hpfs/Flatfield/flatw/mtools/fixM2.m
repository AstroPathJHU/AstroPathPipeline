function d = fixM2(path, sample)
%%--------------------------------------------------------------------
%% Fix all filenames that were created due to an error.
%% In these cases the original .im3 file has been truncated,
%% it exists but cannot be used. The Vectra system then
%% re-wrote te file, but padded the filename with _M2.
%% Here we do two things: if there is an _M2 file, we first
%% delete the file with the short length, then rename the file.
%%
%% Alex Szalay, Baltimore 2018-07-13
%%
%% Usage: fixM2('F:\Clinical_Specimen')
%%    or  fixM2('F:\flatw');
%%--------------------------------------------------------------------
    %
    d = dir([path '\' sample '\**\*_M2.im3']);
    %
    if (numel(d)==0)
        fprintf(1,' fixM2: No *_M2.im3 files found\n');
        return
    end
    %
    for i=1:numel(d)
        %
        f1 = [d(i).folder '\' d(i).name];
        f2 = replace(f1,'_M2','');
        %
        d(i).name
        if (exist(f2))
            delete(f2, [f2 '.old']);
        end
        fprintf(1,' fixM2: Renamed %s => %s\n',f1,f2);
        movefile(f1,f2);
        %
    end
    %
end