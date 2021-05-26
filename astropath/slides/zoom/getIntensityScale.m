function q = getIntensityScale(C)
%%-----------------------------------------------------------------
%% Get the max intensity to be used for 
%% the scaling of the images. We get the
%% 99th percentile iof the first 7 layers
%% (excluding AF), and then take the 98th 
%% percentile of these. We map this value 
%% to 255 when we convert the layers to uint8.
%%
%% 2020-06-27   Alex Szalay
%%-----------------------------------------------------------------
    %
    logMsg(C,'getLayersRaw');
    %    
    F    = [];
    fmax = [];
    if (exist(C.tiffpath)~=7)
        msg = sprintf('ERROR: image path %s missing',path);
        logMsg(msg,1);
        return
    end
    %
    N  = numel(C.H.n);
    %q  = zeros(1,N);
    %ig = zeros(1,N);
    for n=1:N
        o = nextLayer(C,n);
        fprintf('[%04d], %d, %d, %s\n',n,...
            o.status,numel(size(o.a)),C.H.file{n});
        if(o.status==1)
            F{n} = o.a;
            o.a  = o.a(:,:,1:7);  
            q(n) = prctile(o.a(:),[99]);
        else
            F{n}  = [];
            %ig(n) = 0;
            q(n)  = NaN;
        end
    end
    fmax = prctile(q,[98]);    
    fprintf('fmax = %f \n',fmax);
    C.F = F;
    %C.ig = ig';
    %{
    %------------------------------
    % rescale and convert to uint8
    %------------------------------
    for n=1:numel(C.H.n)
        if( ig(n)==1)
            F{n}  = im2uint8(F{n}/fmax);
        end
    end    
    %}
    %
    s = sprintf('getLayers finished, %d images read',...
        numel(C.H.n));
    logMsg(C,s);
    %
end
    N  = numel(C.H.n);
    for n=1:N
        q(n) = nextLayer(C,n);
        fprintf('[%04d],%s\n',n,C.H.file{n});
    end
    fmax = prctile(
    fprintf('[%04d],%s\n',n,C.H.file{n});
    %
end


function fmax = nextLayer(C,n)
%%-------------------------------------
%% Read the next field's image layers.
%% Set the status=0 if the tiff file is missing
%%
%% 2020-06-25   Alex Szalay
%%-------------------------------------
    %
    f = fullfile(C.tiffpath,replace(C.R.file{n},'.im3',C.tiffext));
    try
        info = imfinfo(f);
        for i=1:8
            a(:,:,i)   = imread(f,i,'Info',info);
        end
        fmax = prctile(a(:),[99]);
    catch
        fmax = NaN;
        s = sprintf('ERROR: cannot read %s',f);
        logMsg(C,s,1);
        return
    end
    %
end