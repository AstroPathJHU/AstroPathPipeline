function scan = getScan(root, samp)
%%--------------------------------------------------------
%% determine the latest scan version from the folder names
%% returns the string as used as part of the filename
%%
%% Alex Szalay, Baltimore, 2018-09-24
%%--------------------------------------------------------
    %
    scn =  [root '\' samp '\im3\Scan*'];
    d = dir(scn);
    for i=1:numel(d)
        s(i) = str2num(replace(d(i).name,'Scan',''));
    end
    scan = d(s==max(s)).name;
    %
end
