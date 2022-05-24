%% getset
%% Created by: Benjamin Green
%% ---------------------------------------------
%% Description
% initialize the variables for the mkindvphenim funtion
%% ---------------------------------------------
%%
function [Image,expr] = getset(Markers,imageid)
%
Image.mossize = 50;
expr.namtypes = Markers.expr;
%
Image.ds = imageid.ds;
%
[a, ~] = ismember(Markers.all, expr.namtypes);
Image.ExprCompartment = Markers.Compartment(a);
a = find(a) + 1;
%
expr.layers = a;
%
Image.all_lineages = [Markers.lin, Markers.add];
%
[a, ~] = ismember(Markers.all, Image.all_lineages);
Image.Compartment = Markers.Compartment(a);
a = find(a) + 1;
%
Image.layer = a;
%
end
