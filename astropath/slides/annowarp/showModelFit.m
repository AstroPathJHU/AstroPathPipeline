function showModelFit(M)
    figure(7);
    %
    n=2;
    m=2;
    k=1;
    subplot(n,m,k);
        showModelX(M,78);
        k = k+1;
    subplot(n,m,k);
        showModelX(M,130);
        k = k+1;
    subplot(n,m,k);
        showModelY(M,40);
        k = k+1;
    subplot(n,m,k);
        showModelY(M,70);
        k = k+1;
    shg
    %
end