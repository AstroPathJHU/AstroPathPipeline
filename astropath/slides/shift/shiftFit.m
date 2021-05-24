function [fitresult, gof] = shiftFit(C, x, y, z, txt, plotme, samp)
%%-----------------------------------------------------
%% fit the shift solution to the original coordinates
%%
%% Alex Szalay, 2018-10-14
%%-----------------------------------------------------
    %
    logMsg(C,'shiftFit');
    %
    % reformat the data
    %
    [xData, yData, zData] = prepareSurfaceData( x, y, z );
    %
    % Set up fittype and options.
    %
    ft = fittype( 'poly11' );
    %
    % Fit model to data.
    %
    [fitresult, gof] = fit( [xData, yData], zData, ft );
    %
    % Plot fit with data.
    %
    if (plotme==1)
        %
        h = plot( fitresult, [xData, yData], zData );
        legend( h, txt, 'shift vs. x, y', 'Location', 'NorthEast' );
        xlabel x
        ylabel y
        zlabel z
        grid on
        if (strcmp(txt,'dx'))
            view(0,0);
        else
            view(90,0);
        end
        %
    end
    %
end