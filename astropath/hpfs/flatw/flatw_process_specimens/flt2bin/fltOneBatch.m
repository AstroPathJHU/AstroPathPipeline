function fltOneBatch(tbl3, p1, fnms)
%%
% run the average coding on a list of images
% if the averaging images fail, 
% delete all the mean and csv images
% 
% p1 = fully qualified .bin output image name
% tbl3 = table with sample names, scanpaths, and batchIDs for the
% batch
% fnms = cell array of mean.flt files for averaging
%%
onms = p1;
%
f1 = fullfile(fnms{1}.folder,fnms{1}.name);
f2 = replace(f1,'.flt','.csv');
nn  = csvread(f2);
k = nn(2);
h = nn(4);
w = nn(3);
%
%try
    mean2flat(onms,[fnms{:}],100,k, h, w);
%{
catch
    nm = [fnms{:}];
    nm = strcat({nm(:).folder},'\',{nm(:).name});
    %delete(nm{:})
    nm = replace(nm,'.flt','.csv');
    %delete(nm{:})
end
%}
end
%