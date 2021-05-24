function getFlux(C,A,varargin)
%%----------------------------------------------------
%% will loop through all the layers in a given image
%%
%% 2020-08-06  Alex Szalay
%%----------------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    m = getMask(C,A,opt);
    %
    for i=1:8
        b   = imgaussfilt(A.img{i},20);
        m1  = mean(b(m));
        m2  = median(b(m));
        out = sprintf('%d,%d,%d,%d,%d,%d,%f,%f',C.project,C.cohort,...
            A.cinfo.ncore(1),A.cinfo.TMA(1),A.cinfo.BatchID,i,m1,m2);
        if (~isnan(m1) & ~isnan(m2))
            if (C.opt==0)
                dlmwrite(C.fout,out,'-append','delimiter','','newline','pc');           
            else
                fprintf('%s\n',out);
            end
        else    
                fprintf('%s\n',out);
        end
    end
end