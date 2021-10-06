%% function: getfilenames; Get names of all files in a directory
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/18/2019
%% --------------------------------------------------------------
%% Description
% function takes in a folder location and a Marker struct defined above
% and gets the names of all inform tables
%% --------------------------------------------------------------
%%
function [filenames, err_val]  = getfilenames(wd, Markers)
%
%delete old Results if it exists & create a new results table
%
err_val = 0;
%
filenames = cell(length(Markers.all),1);
nm = cell(1,length(Markers.all));
%
% import file names for all inform file names
%
for i1 = 1:length(Markers.all)
    m = [wd,'\Phenotyped\',Markers.all{i1},'\*cell_seg_data.txt'];
    filenames{i1,1} = dir(m);
    nm{:,i1} = {filenames{i1,1}(:).name};
end
%
%check that all fields are in all inform output
%
for i1 = 2: length(Markers.all)
    a(:,i1-1) = ismember(lower(nm{:,1}),lower(nm{:,i1}));
end
[x,~] = size(a);
ii = zeros(x,1);
%
ii(sum(a,2) == (length(Markers.all) - 1) , 1) = 1;
ii = logical(ii);
%
filenames = filenames{1,1}(ii);
%
% check segmentation 
%
if ~isempty(filenames)
    nm = extractBefore(filenames(1).name,"]_cell");
    %
    if isempty(nm)
        nm = extractBefore(filenames(1).name,"]_CELL");
    end
    %
    % get 1ry segmentation and see if it has proper layers
    %
    wd1 = [wd,'\Phenotyped\',Markers.seg{1},'\'];
    iname = [wd1,nm,']_binary_seg_maps.tif'];
    props = imfinfo(iname);
    if length(props) < 4
        err_val = 11;
        return
    end
    %
    % check 2ry segmentations to see if they have proper layers
    %
    if ~isempty(Markers.altseg)
        for i1 = 1:length(Markers.altseg)
            mark = Markers.altseg{i1};
            wd1 = [wd,'\Phenotyped\',mark,'\'];
            iname = [wd1,nm,']_binary_seg_maps.tif'];
            props = imfinfo(iname);
            if length(props) < 4
                err_val = 12;
                return
            end
        end
    end
end
%
end
