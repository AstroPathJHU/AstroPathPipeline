function cc = saveAffine(C,T,n,samp)
%%---------------------------------------------------
%% convert the affine transformation for the n-th partition to a table
%%
%% Alex Szalay, Baltimore, 2018-08-09
%%---------------------------------------------------
    %
    logMsg(C,'saveAffine');
    %
    % save affine transform table
    %
    c1 = [n,1,1;n,1,2;n,1,3;n,2,1;n,2,2;n,2,3;n,3,1;n,3,2;n,3,3];
    %
    c3 = {'pixels/micron','pixels/micron','zero',...
          'pixels/micron','pixel/micron','zero',...
          'pixels','pixels','one'};
    c4 = {'fx.p10','fy.p10','zero',...
          'fx.p01','fy.p01','zero',...
          'fx.p00','fy.p00','one'};
    cc = table(c1(:,1),c1(:,2),c1(:,3),T(:),c3',c4');
    cc.Properties.VariableNames = {'n','i','j',...
        'value','unit','description'};
    %
end

