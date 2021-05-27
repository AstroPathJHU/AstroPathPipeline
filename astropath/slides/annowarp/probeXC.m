function O = probeXC(C,x,yrange)
    O =[];
    for i=1:numel(yrange)
        y = yrange(i);
        Z = xcregister(C,100,x,y);
        if (Z.flag==1)
            O = [O;x,y,Z.dx,Z.dy];
        end
    end
end