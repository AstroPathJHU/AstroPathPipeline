function v = getParts(fname)
%%----------------------------------------
%% split the filename into parameters
%%----------------------------------------
    %
    f1 = replace(fname,'.','-');
    a = strsplit(f1,'-');
    a = a(2:5);
    %
    for i=1:numel(a)
        v(i) = str2num(a{i}(2:end));
    end
%     if (strcmp(a{3},'Q')>0)
%         q = '';
%     else
%         q = a{3}(2:end);
%     end
%     %
    %
end