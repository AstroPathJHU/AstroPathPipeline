function tracePath(out,varargin)
    %
    scale = 1 ;
    tt = '-';
    c = {'b','r'};
    if (nargin>=2)
        tt = varargin{1};        
    end
    %
    hold on
    %
    % draw positive areas first
    %
    if (numel(out)==0)
        fprintf('empty\n');
        return
    end
    %
    area = clipper(out,0);
    %
    for i=1:length(out)
        if (area(i)>0)
            plot(double([out(i).x;out(i).x(1)])/scale,...
                 double([out(i).y;out(i).y(1)])/scale,...
                 tt,'Color',c{2});
        end
     end
    %
    % holes second
    %
    for i=1:length(out)
        if (area(i)<0)
            plot(double([out(i).x;out(i).x(1)])/scale,...
                 double([out(i).y;out(i).y(1)])/scale,...
                 tt,'Color',c{1});
        end
    end
    %
    axis equal
    %axis([60000 100000 10000 45000]);
    box on
    shg
    %
end