function showPath(out,varargin)
    %
    scale = 1 ;
    c = {'w','r'};
    if (nargin>1)
        c(2) = varargin(1);
    end
    %
    hold on
    %
    % draw positive areas first
    %
    if (numel(out)==0)
        return
    end
    %
    area  = clipper(out,0);
    for i=1:length(out)
         if (area(i)>0)
             fill(double(out(i).x)/scale,double(out(i).y)/scale,c{2});
         end
     end
    %
    % holes second
    %
    for i=1:length(out)
        if (area(i)<0)
            fill(double(out(i).x)/scale,double(out(i).y)/scale,c{1});
        end
    end
    %
    axis equal
    %axis([60000 100000 10000 45000]);
    box on
    shg
    %
end