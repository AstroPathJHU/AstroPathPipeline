% h=surfir(x,y,z,s,opt)
%
% Surface plot based on irregularly spaced data points.
% x,y,z are coordinate vectors of the same length.
% s (default 0.5) is the shrink factor in the Matlab function 'boundary' 
%   (type 'help boundary' for details)
% opt is a list of options to be passed to the 'trisurf' function 
%   (type 'help trisurf' for details)
function h=surfir(x,y,z,s,opt)
%% default parameters
if (nargin<4)||isempty(s)                                                   % no shrink factor provided
   s=0.5;                                                                   % default value
end
if nargin<5                                                                 % no options provided
   opt={'FaceColor','interp','edgecolor','none'};                           % default
end
%% Remove duplicate data points
[xy,ind] = unique([x,y],'rows');
z=z(ind);
x=xy(:,1);
y=xy(:,2);
%% triangulate data points
dt = delaunayTriangulation(x, y);                                           % Delaunay triangulation
x=dt.Points(:,1);
y=dt.Points(:,2);
%% find enclosing boundary
k=boundary(x,y,s);                                                          % define boundary enclosing all data points
c=[k(1:end-1),k(2:end)];                                                    % constraints
dt.Constraints=c;                                                           % add constraints
io = dt.isInterior();                                                       % triangles which are in the interior of domain
tri=dt.ConnectivityList;                                                    % read triangles
tri=tri(io,:);                                                              % use only triangles in the interior
%% plot
h=trisurf(tri,x,y,z,z,opt{:});                                              % plot triangles and interpolate colors
end
