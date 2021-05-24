function q = getAttributes(a,n)
%%--------------------------------------
%% get the Attributes for a given Layer
%%
%% Alex Szalay, Baltimore, 2019-02-19
%%--------------------------------------
    %
    samp = 0;
    p = a.Attributes;
    r = a.Regions;
    cc = dec2hex(str2num(p.LineColor),6);
    cc = string([cc(5:6),cc(3:4),cc(1:2)]);
    %
    %nreg = numel(r.Region);
    poly ={'poly'};
    q = table(samp,n,string(p.Name),cc,strcmp(p.Visible,'True'),poly);
    q.Properties.VariableNames = {'sampleid','layer','name',...
        'color','visible','poly'};
    %    
end
