%% transferComponents
% transfer the components from a source antibody folder to a component
% tiffs folder
%
function transferComponents(inspath)
    %
    sor = [inspath,'\*component_data.tif'];
    des = [inspath, '\..\..\Component_Tiffs'];
    [C] = transferfls(sor,des);
    %
end