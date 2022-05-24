%% mkfigs
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/03/2018
%% --------------------------------------------------------------
%% Description
%%% make the pie chart/ heatmap figures
%% --------------------------------------------------------------
%%
function mkfigs(d,Markers,imageid, mycol)
%
% This section will run for expression markers up to 15, after that the
% code will just skip this portion, if more expression markers are used one
% would need to either make a new page at every 15 or resize the pie
% charts at 'Positions'
%
if length(Markers.expr) > 15
    return
end
%
% create a color container to help specify colors in the pie charts; all
% additional markers added by acceptable coexpression will be black
%
for i1 = 1:length(Markers.lin)
    cmapvar{i1} = mycol.lin(i1,:);
end
%
for i2 = 1:length(Markers.add)
    i1 = i1 + 1;
    cmapvar{i1} = [0 0 0];
end
%
cmapvar{i1 + 1} = [0 0 1];
%
marks = [Markers.lin, Markers.add, 'Other'];
colorcont = containers.Map(marks, cmapvar);
%
% specify the positions for the pie charts:
%
% left shift = .18
% down shift = .32
% Grid:
% 1 - 3 - 7 - 10 - 13
% 2 - 4 - 8 - 11 - 14
% 5 - 6 - 9 - 12 - 15
%
Positions = {...
    [0.04 0.64 0.1 0.2], [0.04 0.35 0.1 0.2],...
    [0.22 0.64 0.1 0.2], [0.22 0.35 0.1 0.2],...
    [0.04 0.06 0.1 0.2], [0.22 0.06 0.1 0.2],...
    [0.40 0.64 0.1 0.2], [0.40 0.35 0.1 0.2], ...
    [0.40 0.06 0.1 0.2], [0.58 0.64 0.1 0.2], ...
    [0.58 0.35 0.1 0.2], [0.58 0.06 0.1 0.2],...
    [0.76 0.64 0.1 0.2], [0.76 0.35 0.1 0.2],...
    [0.76 0.06 0.1 0.2]};
%
% get data types into a conditional format that is easier to work with for
% the pie charts
%
data{1} = d.fig;
types{1} = ['All Markers (n = ',num2str(height(d.fig)),')'];
for i1 = 1:length(Markers.expr)
    m = Markers.expr{i1};
    l = lower(Markers.expr{i1});
    d.fig.(l) = logical(d.fig.(l));
    data {i1 + 1} = d.fig(d.fig.(l),:);
    hh = num2str(height(data{i1+1}));
    types{i1 + 1} = [m,' Expression (n = ',hh,')'];
end
%
% Create figure
%
XX = figure('visible' , 'off');
set(gcf, 'units','normalized','outerposition',[0 0 1 1],...
    'PaperUnits','inches');
XX.PaperPositionMode = 'auto';
%
% add a title to the figure
%
str = extractBefore(d.fname.name,'_cleaned');
str = ['Image QC Output for: ', str];
annotation('textbox', [.004 .8 .1 .2], 'String',str,'FontSize',14,...
    'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on',...
    'Interpreter','none')
%
% Make pie charts: on the matlab help page - Label Pie Chart With Text and
% Percentages - can also provide additional changes to axis
%
for i2 = 1:length(data)
    %
    % get Phenotypes in this data type
    %
    m = unique(data{i2}.Phenotype);
    explode = ones(length(m),1)';
    %
    if ~isempty(m)
        %
        % get colors for markers in this data type
        %
        cmap = zeros(length(m), 3);
        %
        for i1 = 1:length(m)
            m1 = m{i1};
            c = colorcont(m1);
            cmap(i1,:) = c;
        end
        %
        % create graph at Position designated
        %
        p(i2) = axes('Position',Positions{i2});
        %
        pie(p(i2), categorical(string(data{i2}.Phenotype(:))),...
            explode);
        %
        title(p(i2), types{i2})
        %
        set(get(gca,'title'),'Position',[0.1 1.5 1])
        %
        colormap(p(i2),cmap);
    end
end
%
% Make a heat map
%
% create a color scale
%
r = [ones(1,30)', linspace(1,0,30)', linspace(1,0,30)'];
b = [linspace(0,1,30)', linspace(0,1,30)',ones(1,30)'];
cmap = vertcat(b,r);
%
% get number of unique phenotypes to make heatmap bars for
%
m = unique(d.fig.Phenotype);
marks = [Markers.all,Markers.add,'Other'];
m1 = ismember(marks,m);
m = marks(m1);
%
% set positions for the heatmap; the rest of the heatmap should shift
% appropriately if these values are changed
%
% ls: left side of the first heatmap on the chart
% szw: the width of the heatmap bars
% szh: the height of the heatmap bars
% b: the bottom of the heatmap
%
ls = .84;
szw = .075;
szh = .75;
b = .13;
%
% a check that will separate the figures into different image files if the
% data types need this
%
if (length(data) > 6 && length(m) > 3) ||...
        length(data) > 9 || ...
        (length(m) > 6 && length(data)~=1)
    print(XX, strcat(imageid.outfull,...
        'cleaned_phenotype_pie_charts.tif'),'-dtiff','-r0');
    close all
    XX = figure('visible' , 'off');
    set(gcf, 'units','normalized','outerposition',[0 0 1 1],...
        'PaperUnits','inches');
    XX.PaperPositionMode = 'auto';
    %
    % add title to figure
    %
    annotation('textbox', [.004 .8 .1 .2], 'String',str,'FontSize',14,...
        'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on',...
        'Interpreter','none')
    %
    % move the heatmap to the middle of the page
    %
    ls = .84 - 2 * szw;
    szw = .075;
    szh = .75;
    b = .13;
    %
end
%
% set the positions of the heatmap bars for each phenotype
%
Pos1 = [ls 0.13 szw szh];
%
for i1 = 1: length(m)
    ps1 = Pos1(i1,1) - szw;
    Pos1 = [Pos1;ps1 b szw szh];
end
Pos1 = flip(Pos1);
%
% create the heatmaps
% done in reverse order to help with formatting
%
for i1 = length(m):-1:1
    x1 = d.fig(strcmp(d.fig.Phenotype,m{i1}),:);
    %
    %find log intensity of each marker in total cell
    %
    x = [];
    for i2 = 1:length(Markers.Opals)
        intes = ['MeanEntireCell',num2str(Markers.Opals(i2))];
        var = (log(x1.(intes) + .001));
        x(i2,:) = var;
    end
    x = flip(x);
    %
    axes('Position',Pos1(i1,:));
    h(i1) = heatmap(x, 'Colormap', cmap,'ColorbarVisible', 'off',...
        'GridVisible', 'off');
    % h(i1).XLabel = m{i1};
end
%
% set the colorbar
%
h(length(m)).ColorbarVisible = 'on';
%
% set the y axis to the Opal names
%
Op = flip(Markers.Opals);
Op = num2str(Op);
Op = [repmat('Opal ',length(Op),1), Op];
Op = cellstr(Op);
h(1).YData = Op ;
%
% hide the cell numbers that show on the bottom of the heatmap
%
str = ' ';
ps(1,1) = Pos1(1,1) - .03;
ps(1,2) = b - .08;
ps(1,3) = length(m) * szw + .06;
ps(1,4) = b - ps(1,2) - .002;
%
annotation('textbox', ps, 'String',str,...
    'BackgroundColor' , 'white','EdgeColor','white');
%
% set the x axis to the marker names
%
for i1 = 1: length(m)
    m1 = m{i1};
    %[0.465 0.13 0.075 0.75]
    ps =  Pos1(i1,:);
    ps(1,1) = Pos1(i1,1) + szw/5;
    ps(1,2) = b - .04;
    ps(1,3) = .05;
    ps(1,4) = .03;
    annotation('textbox', ps, 'String', m1,'FontSize',14,...
        'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on');
end
%
% display which Opals target which ABs under the xaxis labels
%
Op = flip(Op);
Op1 = [];
for i1 = 1:length(Markers.all)
    A = Op{i1}(~isspace(Op{i1}));
    Op1 = [Op1,Markers.all{i1},': ',A,'; ',];
end
%
Op1 = ['Note: ',Op1];
%
ps(1,1) = Pos1(end,1) - (length(Markers.Opals) * .05);
%
%ps(1,1) = Pos1(1,1) - .05;
ps(1,2) = .055;
annotation('textbox', ps, 'String', Op1,'FontSize',14,...
    'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on');
%
% make a title for the heatmap
%
str = 'Heatmap of Phenotype vs Opal Intensities in Cells';
if length(m) <= 2
    ps(1,1) = Pos1(1,1) - szw/2;
else
    sz = floor((length(m)-1)/2);
    ps(1,1) = Pos1(sz,1) + szw/5;
end
%
ps(1,2) = Pos1(1,2) + szh + .01;
annotation('textbox', ps, 'String', str,'FontSize',16,...
    'BackgroundColor' , 'white','EdgeColor','white','FitBoxToText','on');
%
print(XX, strcat(imageid.outfull,...
    'cleaned_phenotype_data.tif'),'-dtiff','-r0');
close all
%
end