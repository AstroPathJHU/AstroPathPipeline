function C=plotAll()
    %
    close all
    C = plotFlux('rawFlux',1);
    plotFlux('normFlux',2);
    plotFlux('flatFlux',3);
    plotMeanFlux(4);
    plotBox(C,1,5);
    plotBox(C,2,6);
    %
end