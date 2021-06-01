function q = path2char(p)
%%-------------------------------------------------------
%% take a single clipper Path object 
%% and write the coordinates to a string.
%%-------------------------------------------------------
    %
    x = [p.x;p.x(1)];
    y = [p.y;p.y(1)];
    z = [x';y']';
    q = ['(', char(join(join(string(z))',',')),')'];
    %
end
