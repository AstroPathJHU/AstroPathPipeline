function C=tweakM211(C)
%%-----------------------------------------------------
%% try to fix the problem in M211 and in M118 due to Yposition=0.
%% We cannot shift the images as teh grid is then not aligned
%% We must shift at the cross correlation point in makeWarp
%%
%% 2020-07-15   Alex Szalay
%%-----------------------------------------------------
    %
    % shift the aimg in the xcorr by 900 pixels
    %
    C.qshifty = 900;
    %
    % also shift the vertices in C.PV by the same amount
    %
    %y = C.PV.y - C.qshift;
    %
end