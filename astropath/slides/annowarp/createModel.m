function C = createModel(C)
%%-------------------------------------------------------
%% Bind all the pieces together to create the model fit
%%
%% 2020-07-15   Alex Szalay
%%-------------------------------------------------------
    %
    C.Y.M = makeModel(C);
    C.Y.R = fitModel(C);
    C.Y.M = addModels(C);
    %    
    C.Y.F = getFlats(C);    
    C.Y.M = addFlats(C);
    %
end