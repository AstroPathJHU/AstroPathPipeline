function T = writeZoomList(C)
%%--------------------------------------------------
%% create a table of all the images to be loaded
%% into the database, and write it to a CSV file
%%--------------------------------------------------
    %
    t0 = clock();
    logMsg(C.samp,'writeZoomList');
    %
    T = [];
    for n=0:9
        t = getCsvList(C,n);
        T = [T;t];
    end
    %
    ff = [C.zoompath,'zoomlist.csv'];
    writetable(T,ff);
    %
    s = sprintf('writeZoomList finished in %f sec',etime(clock(),t0));
    logMsg(C.samp,s);
    %    
end


function t = getCsvList(C, zoom)
%%----------------------------------------------
%% get table of all the images at a given zoom 
%%----------------------------------------------
    %
    ff = [C.zoompath,sprintf('%d',zoom),'\*.png'];
    d  = dir(ff);
    if (numel(d)>0)
        for i=1:numel(d)
            %
            %[z(i),m(i),q{i}] = getParts(d(i).name);
            v  = getParts(d(i).name);
            z(i) = v(1);
            m(i) = v(2);
            x(i) = v(3);
            y(i) = v(4);
            %
            if (z(i)~=zoom)
                fprintf('ERROR: zoom level mismatch\n');
                return
            end
            p{i} = C.samp;
            f{i} = [d(i).folder,'\',d(i).name];
            %
        end
    end
    %
    
    t =table(p',z',m',x',y',f');
    t.Properties.VariableNames = {'sample','zoom','marker',...
        'x','y','name'};
    %
end
