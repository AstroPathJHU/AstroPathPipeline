function showTileGrid(T)
%%------------------------------------
%% display the two grids overlaid
%%------------------------------------
    %
    close all
    %
    drawPatches(T);
    ax = gca;
    ax.YDir = 'reverse';
    axis equal
    axis([8350,8420,4180,4250]);
    box on
    %    
    shg
    %
    %
end

function drawPatches(T)
    nx = numel(unique(T.x))/2;
    ny = numel(unique(T.y))/2;
    %
    for i=1:nx
        ix = 1+2*(i-1);
        for j=1:ny
            iy = 1+2*(j-1);
            %
            x = T.x(ix:ix+1,iy:iy+1);
            y = T.y(ix:ix+1,iy:iy+1);
            p = [1,2,4,3];
            dx = T.dx(ix:ix+1,iy:iy+1);
            dy = T.dy(ix:ix+1,iy:iy+1);
            %{
            patch(x(p),y(p),'b','EdgeColor','b','FaceColor','none',...
                'FaceAlpha',0.4)
            %}
            patch(x(p)+dx(p),y(p)+dy(p),'g','FaceColor','g',...
                'FaceAlpha',0.25)
            
        end
    end
    %
end