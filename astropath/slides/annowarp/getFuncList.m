function t = getFuncList()
    %
    d = dir('*.m');
    c =struct2cell(d);
    t = table(c(1,:)');
    t.Properties.VariableNames = {'name'};
    %
end