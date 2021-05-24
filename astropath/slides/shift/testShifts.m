function [C] = testShifts(C)
%%------------------------------------------------------------
%% use the spring solution and compute the standard deviation
%% before and after the solution
%%------------------------------------------------------------
    %
    logMsg(C,mfilename);
    %
    close all
    %
    % update the positions from the displacements
    % show only those where the mse > 10,000
    %
    iz = (abs(C.S.dx)<C.limit & abs(C.S.dy)<C.limit) & C.S.mse1>10000;
    iz = iz & C.S.p1<C.S.p2;
    %
    % ignore fields from skipped partitions, just in case
    %
    iq = [];
    for n=1:numel(C.W)
        if(C.W{n}.skip==1)
            iq = [iq,C.W{n}.g];
        end
    end
    %
    u = unique([find(ismember(C.S.p1,iq))',find(ismember(C.S.p2,iq))']);
    iz(u) = 0;
    %
    % create the final selection of overlapping fields
    %
    p1 = C.S.p1(iz);
    p2 = C.S.p2(iz);
    %
    % get the precise shift
    %
    dx1 = C.S.dx(iz);
    dy1 = C.S.dy(iz);
    %
    dx2 = C.V.ZX(p1) - C.V.ZX(p2)-dx1;
    dy2 = C.V.ZY(p1) - C.V.ZY(p2)-dy1;
    %
    C.G = table(p1,p2,dx1,dy1,dx2,dy2);
    C.G.Properties.VariableNames={'p1','p2','dx1','dy1','dx2','dy2'};
    %
    C.sigma = table(string(C.samp),std(C.G.dx1),std(C.G.dy1),...
        std(C.G.dx2),std(C.G.dy2));
    C.sigma.Properties.VariableNames={'sample','sx1','sy1','sx2','sy2'};
    %
    %
    % do the graphics if the flag is set
    %
    if (C.flag>0)
        showResults(C);
    end
    %
end





