function setLog(val)
%%-----------------------------------------------
%% turn hard logging on or off
%%     setLog('on') will turn the hard log on
%%      setLog('off') will turn it off
%%
%% Alex Szalay 2020-11-6
%%-----------------------------------------------
global logctrl
    %
    if (strcmp(val,'on')>0)
        logctrl=1;
    else
        logctrl = 0;
    end
    %
end