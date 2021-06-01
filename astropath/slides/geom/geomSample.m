function [C] = geomSample(root,samp,varargin)
%%---------------------------------------------------------
%% extract the relevant geometries from the InForm layers
%% and convert them into WKT polygon format
%%
%% Alex Szalay, Baltimore, 2019-03-07
%%---------------------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    C = getConfig(root,samp,'geom');
    logMsg(C,'geomSample started',1);
    %
    C = readMetadata(C);
    if (C.err==1)
        return
    end
    %------------------------------------------------------------
    % compute the boundary polygons for the Fields and Tumor
    %------------------------------------------------------------
    [T,F] = getBoundaries(C);
    %
    if (opt==-1)
        return
    end
    %
    % write the output into dbload
    %
    out = [C.dbload,'\',C.samp];
    try
        writetable(T,[out,'_tumorGeometry.csv']);
    catch
        fprintf('No Tumor booundaries detected\n');
    end
    %
    try
        writetable(F,[out,'_fieldGeometry.csv']);
    catch
        fprintf('No Field booundaries detected\n');
    end
    %
    C.T = T;
    C.F = F;
    %
    logMsg(C,'geomSample finished');
    %
end

