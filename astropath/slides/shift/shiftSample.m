function C = shiftSample(root, samp, varargin)
%%----------------------------------------------------------------
%% Determine the optimal shifts for a given sample.
%% Writes the samp_fields.csv to the dbload directory if opt==0
%% root1 is the path to the cohort, e.g. '\\BKI02\E$\Clinical_Specimen'
%% root2 is the path to the flatw images '\\BKI02\F$\flatw'
%% sample is the name of the sample, e.g. 'M15_1'
%%
%%   C = runShift('X:\Clinical_Specimen','Y:\flatw','M18_1');
%%
%% Alex Szalay, Baltimore, 2018-08-09
%%----------------------------------------------------------------
    %
    C = getConfig(root,samp,'shift');
    logMsg(C,'shiftSample');
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    % read the relevant data from the csv files
    %
    C = readMetadata(C);
    C.S = C.A;
    if (C.err>0)        
        return
    end
    %
    % set graphics option
    %
    C.opt  = opt;
    C.flag = C.opt;
    %
    if (opt==-1)
        return
    end
    %
    % copy R->H, and calculate the fiducial origin of the images
    %
    C.H = C.R;
    %
    C.H.ix = floor(C.pscale*C.H.x);
    C.H.iy = floor(C.pscale*C.H.y);
    %
    if (opt==-2)
        return
    end
    %
    % determine if the graph has any partitions
    %
    C = solvePartitions(C);
    %
    % create empty arrays for partition-wise updates
    %
    C.V = table(C.H.n);
    C.V.Properties.VariableNames = {'n'};
    C.T = [];    %
    if (opt==-3)
        return
    end
    %    
    for n=1:numel(C.W)
        %
        C = solveLaplace(C,n);
        g = C.W{n}.g;
        %
        try
            %
            % get the affine transformation for each partition
            %
            fx = shiftFit(C,C.H.x(g),C.H.y(g),C.Z{n}.X,'dx',0,C.samp);
            fy = shiftFit(C,C.H.x(g),C.H.y(g),C.Z{n}.Y,'dy',0,C.samp);
            T  = [fx.p10,fy.p10,0;fx.p01,fy.p01,0;fx.p00,fy.p00,1]';
            %
            % fit was ok, continue
            %
            C.W{n}.skip = 0;
            aa = saveAffine(C,T,n,C.samp);
            C.T = [C.T;aa];
            %
            logMsg(C,sprintf('Fit partition %d, size %d',...
                n, numel(find(C.W{n}.g))) );
            %
            % update the final positions, into linear arrays
            %
            C.H.px(g) = C.H.ix(g) + C.Z{n}.X - T(3,1);
            C.H.py(g) = C.H.iy(g) + C.Z{n}.Y - T(3,2);
            %
            % save the displacements for plotting
            %
            C.V.ax(g) = feval(fx,C.H.x(g),C.H.y(g));
            C.V.ay(g) = feval(fy,C.H.x(g),C.H.y(g));
            C.V.ZX(g) = C.Z{n}.X;
            C.V.ZY(g) = C.Z{n}.Y;
            %
            % compute the discrete grid for the fields
            %
            C = quantizeGrid(C,n);
            %
        catch
            %
            % not enough points to fit, skip the partition
            %
            C.W{n}.skip = 1;
            C.H.gc(C.W{n}.g) = 0;            
            logMsg(C,sprintf('Skip partition %d, size %d',...
                n, numel(find(g))));
        end
        %
    end
    %
    if (opt==-4)
        return
    end
    %
    %  show test result only if flag is set
    %   
    C = testShifts(C);
    %
    % adjust the H table with the offsets
    %
    xoffset = C.xposition*C.pscale;
    yoffset = C.yposition*C.pscale;
    %
    C.H.gx  = C.R.gx;
    C.H.gy  = C.R.gy;
    %
    C.H.px  = C.H.px  -xoffset;
    C.H.py  = C.H.py  -yoffset;
    C.H.mx1 = C.H.mx1 -xoffset;
    C.H.my1 = C.H.my1 -yoffset;
    C.H.mx2 = C.H.mx2 -xoffset;
    C.H.my2 = C.H.my2 -yoffset;
    %
    % write the output files if opt==0
    %
    if (C.opt==0)
        writeShifts(C);
    end
    %
end


function writeShifts(C)
%%------------------------------------------------------------------
%% write the results of the stitching solution to sample\dbload
%%
%% Alex Szalay, Baltimore, 2019-02-03
%%------------------------------------------------------------------
    %
    logMsg(C,mfilename);
    %
    % save the transformation matrices and the R table
    %
    writetable(C.T,[C.dbload,sprintf('%s_affine.csv',C.samp)]);   
    writetable(C.H,[C.dbload,sprintf('%s_fields.csv',C.samp)]);
    writetable(C.sigma,[C.dbload,sprintf('%s_sigma.csv',C.samp)]);
    %
end


