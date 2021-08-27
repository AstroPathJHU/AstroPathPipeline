%%
function p2 = tryCATCHwritetable(pqt, pqf)
%%
% write a table with error catching. Since multiple functions and users may
% access tables it may not always be allowed to open save over the current
% process tables.
%
%%
    %
    p2 = 0;
    tt = 0;
    wpqf = replace(pqf, '\','\\');
    %
    while p2 == 0 && tt < 10
        try
            writetable(pqt, pqf);
            p2 = 1;
        catch
        end
        if ~p2
            pause(10*60)
            fprintf(['Attempt ',num2str(tt + 1),' to write to ',...
                wpqf,' FAILED...\n   Attempting to transfer again in ',...
                '10 MINUTES...\n'])
            tt = tt + 1;
        end
    end
    %
end