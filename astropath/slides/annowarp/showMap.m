function showMap(T)
    %
    close all
    C = sign(T.y-8390);
    C = C-sign(T.y-8392);
    C=C/2;
    %
    surf(T.x,T.y,T.y+T.dy,C,'FaceAlpha','0.75');
    hold on
    %surf(T.x,T.y,0*T.y+8335,C,'FaceAlpha','1'); 
    %surf(T.x,0*T.y+8360,T.y-T.dy,C,'FaceAlpha','1'); 
    axis([8260 8330 8360 8430 8335 8410]);
    %
    xlabel('ax');
    ylabel('ay');
    zlabel('qy');
    box on
    view(97.5,9.4);
    %
end