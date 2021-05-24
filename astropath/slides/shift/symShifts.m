function U = symShifts(C)
%%--------------------------------------------------
%% make the shift matrix symmetric. From each pair 
%% always pick the overlap that has the lower dv value
%%
%% Alex Szalay, Baltimore 2018-08-09
%%--------------------------------------------------
    %
    logMsg(C,mfilename);
    %    
    % make the shift table symmetric
    %
    ilo = C.S.p1<C.S.p2;
    A = C.S(ilo,:);
    B = C.S(~ilo,:);
    %
    A = sortrows(A,[2,3]);
    B = sortrows(B,[3,2]);
    B = B(:,[1,3,2,(4:11)]);
    A = A(:,[1:3,7:9,11,4]);
    B = B(:,[1:3,7:9,11]);
    %
    A.Properties.VariableNames = {'n1','p1','p2',...
        'dx','dy','sc','dv','code'};
    B.Properties.VariableNames = {'n2','p1','p2',...
        'dx2','dy2','sc2','dv2'};
    %
    % join the two tables
    %
    U = join(A,B);
    %
    % resolve the differences across the diagonal,make it symmetric
    %
    ix = U.dv<U.dv2;
    U.dx2(ix) = -U.dx(ix);
    U.dy2(ix) = -U.dy(ix);
    U.dv2(ix) =  U.dv(ix);
    U.sc2(ix) =  1./U.sc(ix);
    U.sc(~ix) =  1./U.sc2(~ix);
    %
    % clean up the table
    %
    N   = numel(U.n1);
    U.n = (1:N)';
    U   = U(:,[14,8,2:7]);
    %
end
