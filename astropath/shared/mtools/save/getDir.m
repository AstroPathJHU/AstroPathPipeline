function T = getDir(C)
%%----------------------------------------------------------
%% gets the im3 file info in a directory.
%% parses the sample and coordinates from the
%% filename, and extracts the datenum. Returns
%% an array sorted by the datenum in increasing order
%%
%%----------------------------------------------------------
    %
    path = [C.root,'\',C.samp,'\Im3\',C.scan,'\MSI'];
    p = [path,'\*.im3'];
    d = dir(p);
    %
    % get the sample name
    %
    sample = regexp(d(1).folder,'(M[0-9]+_[0-9]+)','match');
    p1 = '[0-9]+,[0-9]+';
    p2 = '[0-9]+';
    %
    for i=1:numel(d)
        %
        a = regexp(d(i).name,p1,'match');
        b = regexp(a,p2,'match');
        %
        x(i) = double(str2num(b{1}{1}));
        y(i) = double(str2num(b{1}{2}));
        t(i) = posixtime(datetime(d(i).datenum,'ConvertFrom','datenum'));
    end
    %
    C.x = x;
    C.y = y;
    C.t = t;
    %
    T = table(C.x',C.y',C.t');
    T.Properties.VariableNames = {'cx','cy','t'};
    T = sortrows(T,'t');
    %
end

