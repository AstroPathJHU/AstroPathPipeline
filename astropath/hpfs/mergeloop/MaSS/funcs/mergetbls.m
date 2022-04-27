%% function:mergetbls; merge tables function for inform output
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2019
%% --------------------------------------------------------------
%% Description
% function takes in a filename, a data structure with marker information,
% and a directory location
% it generates a *\Tables\ directory and merged inform files
%% --------------------------------------------------------------
%%
function [fData, e_code] = mergetbls(fname, Markers, wd, imall)
%
fData = [];
%
% read in data
%
[C, units, e_code] = readalltxt(fname, Markers, wd);
%
if e_code ~= 0
    return
end
%
% units check
%
if any(~strcmp(units, 'pixels'))
    e_code = 17;
end
%
% select proper phenotypes
%
f = getphenofield(C, Markers, units);
%
% remove cells within X number of pixels in cells
%
d = getdistinct(f, Markers);
%
% removes double calls in hierarchical style
%
q = getcoex(d, Markers);
%
% get polygons from inform and remove other cells inside secondary 
% segmentation polygons
%
a = getseg(q, Markers);
%
% determine expression markers by cells that are in X radius
%
fData = getexprmark(a, Markers);
%
% save the data
%
ii = parsave(fData, fname, Markers, wd, imall);
%
end