%% mkpaths
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% get the top 20 Immune fields from the .mat files in tmp_ForFigureTables
%%% and create the paths necessary for the rest of the code
%% --------------------------------------------------------------
%%
function [charts1, e] = mkpaths(Markers, wd, allimages, doseg)
%
e = 0;
%
% Remove any old ByImage directory
%
m{1} = [wd,'\Phenotyped\Results\QA_QC'];
%
% create new ByImage subdirectories
%
m{2} = [m{1},'\Phenotype'];
m{3} = [m{1},'\Tables_QA_QC'];
mkdir(m{3})
mkdir (m{2},'All_Markers')
%
m{1} = [m{1},'\Lin&Expr_Coex'];
%
for z = 1:length(Markers.all)
    mkdir(m{2}, Markers.all{z})
end
for z = 1:length(Markers.lin)
    mkdir(m{1}, Markers.lin{z})
end
for z = 1:length(Markers.add)
    mkdir(m{1}, Markers.add{z})
    mkdir(m{2}, Markers.add{z})
end
%
layers = length(Markers.Opals) + 2;
%
% get charts; determined based off of Blank, CD8, and Tumor percentages if
% there are more than 20 HPFs
%
try
    charts = dir([wd,'\Phenotyped\Results\tmp_ForFiguresTables\*.mat']);
catch
    e = 11;
    return
end
if isempty(charts)
    e = 11;
    return
end
%
charts1 = charts;
%
try 
    if length(charts1) > 20 && ~allimages
        inc = 1;
        while length(charts1) ~= 20 && inc <= 2
            formatspec = strcat(repmat('%s ',[1,5]),{' '},repmat('%f32 ',[1,10]),...
                { ' %s '},repmat('%f32 ',[1,5]),{' '},repmat('%f32 ',[1,5*layers]),...
                { ' %s '},repmat('%f32 ',[1,5]),{' '},repmat('%f32 ',[1,5*layers]),...
                { ' %s '},repmat('%f32 ',[1,5]),{' '},repmat('%f32 ',[1,5*layers]),...
                { ' %s '},repmat('%f32 ',[1,4]),{' '},repmat('%f32 ',[1,5*layers]),...
                {' '},repmat('%s ',[1,2]),{' '}, repmat('%f32 ',[1,4]),{' '}, ....
                repmat('%s ',[1,2]));
            formatspec = formatspec{1};
            %
            fd = [wd,'\Phenotyped\Results\tmp_ForFiguresTables'];
            nms = {charts(:).name};
            query3 = cell(1,length(charts));
            query2 = cell(1,length(charts));
            parfor i2 = 1:length(charts)
                nm = nms{i2};
                [query2{i2},query3{i2}] = delextrfields(fd,nm,wd,Markers,...
                    formatspec,inc);
            end
            %
            query2 = [query2{:}];
            query3 = [query3{:}];
            %
            query3 = query3(query2);
            charts1 = charts(query2);
            %
            [~,query4] = sort(query3,2,'descend');
            if length(query4) < 20
                a = query4(1:end);
            else
                a = query4(1:20);
            end
            %
            charts1 = charts1(a);
            inc = inc + .25;
        end
    end
catch
    e = 12;
    return
end
%
% check segmentation
%
if ~isempty(charts1) && doseg
    nm = extractBefore(charts(1).name,'cleaned_phenotype');
    %
    % get 1ry segmentation and see if it has proper layers
    %
    wd1 = [wd,'\Phenotyped\',Markers.seg{1},'\'];
    iname = [wd1,nm,'binary_seg_maps.tif'];
    props = imfinfo(iname);
    if length(props) < 4
        e = 13;
        return
    end
    %
    % check 2ry segmentations to see if they have proper layers
    %
    if ~isempty(Markers.altseg)
        for i1 = 1:length(Markers.altseg)
            mark = Markers.altseg{i1};
            wd1 = [wd,'\Phenotyped\',mark,'\'];
            iname = [wd1,nm,'binary_seg_maps.tif'];
            props = imfinfo(iname);
            if length(props) < 4
                e = 13;
                return
            end
        end
    end
end
%
end
