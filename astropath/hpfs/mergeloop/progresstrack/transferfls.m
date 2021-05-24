%% Transfer files between directories fast
%% --------------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hokpins, Baltimore 12/13/2018
%% --------------------------------------------------------------------
%% Description
%%% transfer all files from sorce folder to destination folder using a
%%%parfor loop
%% input
%%% sor: the sorce directory
%%% des: the destination directory
%% --------------------------------------------------------------------
%%
function [C] = transferfls(sor,des)
cfiles = dir(sor);
ii = strcmp({cfiles(:).name}, '.')|strcmp({cfiles(:).name},'..');
cfiles = cfiles(~ii);
C = ~isempty(cfiles);
if C
    if ~exist(des,'dir')
        mkdir(des)
    end
    parfor i3 = 1:length(cfiles)
        sor = [cfiles(i3).folder,...
            '\', cfiles(i3).name];
        try
            movefile(sor, des)
        catch
        end
    end
end
end