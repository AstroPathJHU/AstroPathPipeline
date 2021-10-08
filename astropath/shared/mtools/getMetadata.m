function C = getMetadata(root, samp)
%%--------------------------------------------------------------
%% Use this function to load the metadata previously created
%%
%% 2020-06-11   Alex Szalay
%%--------------------------------------------------------------
    %
    C = getConfig(root,samp,'getMetadata');
    C = readMetadata(C);
    %
end