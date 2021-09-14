function wipeinform(base, slideids)
    %
    f = dir(base);
    %
    for i1 = 1:length(f)
        if ~(contains(slideids, f(i1).name))
            continue
        end
        %
        sor = fullfile(f(i1).folder, f(i1).name, '\inform_data\Phenotyped\Results');
        %
        if exist(sor, 'dir')
            fprintf([replace(sor,'\','\\'),'\n'])
            rmdir(sor, 's')
        end
        %
    end
    %
end