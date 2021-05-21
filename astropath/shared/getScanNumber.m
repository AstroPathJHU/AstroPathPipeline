function snum  = getScanNumber(root, samp)
%%--------------------------------------------------------
%% Determine the latest scan version from the folder names
%% Returns a decimal number
%%
%% Alex Szalay, Baltimore, 2018-09-24
%%--------------------------------------------------------
    %
    scn =  [root '\' samp '\im3\Scan*'];
    d = dir(scn);
    s = [];
    for i=1:numel(d)
        %fprintf('%s %s\n',d(i).folder,d(i).name);
        s(i) = str2num(replace(d(i).name,'Scan',''));
    end
    snum = max(s);
    %
end
