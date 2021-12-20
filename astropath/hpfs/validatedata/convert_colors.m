function [mycol, err_val] = convert_colors(B, err_val)
color_names = {'red','green','blue','cyan', ...
    'magenta','yellow','white','black','orange','coral'};
color_names2 = {'r','g','b','c','m','y','w','k','o','l'};
colors = [eye(3); 1-eye(3); 1 1 1; 0 0 0; 1 .7529 .7961; 0.91 0.41 0.17;];
%
if isa(B.Colors, 'cell')
    [ii,loc] = ismember(B.Colors, color_names);
    [ii1,loc1] = ismember(B.Colors, color_names2);
    ii = ii + ii1;
    loc = loc + loc1;
    %
    if sum(ii) ~= length(B.Colors)
        new_colors = B.Colors(~ii);
        new_colors = replace(new_colors, {'[',']',' '},'');
        new_colors = cellfun(@(x) strsplit(x, ','), new_colors, 'Uni', 0);
        new_colors = cellfun(@str2double, new_colors, 'Uni', 0);
        if any(cellfun(@length, new_colors)~=3)
            err_val = err_val + 2;
            return
        end
        new_colors = cell2mat(new_colors);
        if any(new_colors > 255)
            err_val = err_val + 2;
            return
        end
        loc(~ii) = (length(colors) + 1):...
            ((length(colors)) + (length(B.Colors) - sum(ii)));
        colors = [colors;new_colors];
    end
    %
    %% From https://www.mathworks.com/matlabcentral/fileexchange/46289-rgb2hex-and-hex2rgb:
    mycol.all = [colors(loc,:); 0, 0, 0];
    mycol.hex(:,2:7) = reshape(sprintf('%02X',  round(mycol.all*255).'),6,[]).'; 
    mycol.hex(:,1) = '#';
    %
else
    err_val = err_val + 2;
    return
end
end