function checkLayers(C)
    for i=1:numel(C.F)
        if (size(C.F{i},1)==0)
            fprintf('%d\n',i);
    end
end