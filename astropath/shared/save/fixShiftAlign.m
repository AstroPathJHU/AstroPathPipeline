function S = fixShiftAlign(C)
%%---------------------------------------------------------------
%% change the *_shift.csv file s*_align.csv
%%
%% Alex Szalay, Baltimore, 2019-02-07
%%---------------------------------------------------------------
    %
    f1 = [C.dest C.samp '_shift.csv'];
    f2 = [C.dest C.samp '_align.csv'];
    %
    if (exist(f1)==0)
        return        
    end
    %
    S = readtable(f1,'Delimiter',',');
    %
    % get rid of the corner intersections
    %
    ix = ismember(C.O.tag,[2,4,6,8]);
    %
    S = S(ix,:);
    writetable(S,f2);
    %
    movefile(f1,[f1 '.old']);
    %
end