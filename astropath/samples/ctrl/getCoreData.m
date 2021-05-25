function Z = getCoreData(C,n)
%%-----------------------------------------------------
%% collect the images corresponding to the needed cores
%% from a given control sample (n)
%%
%% 2020-08-04   Alex Szalay
%%----------------------------------------------------
    %
    msg  = [C.ctrl.SlideID{n},' started'];
    logMsg(C,msg);    
    %
    if (n>numel(C.ctrl.TMA))
        msg = 'ERROR: Control sample index out of range';
        logMsg(C,msg,1);
        C.err=1;
        return
    end
    %
    ctrl = C.ctrl(n,:);
    C.samp = ctrl.SlideID{1};
    f = fullfile(C.root,C.samp,'inform_data','Component_Tiffs','*.tif');
    d = dir(f);
    %
    k = 1;
    Z = [];
    for i=1:numel(d)
        %------------------------------------
        % test if the core is on our list
        %------------------------------------
        core = findCore(C,d(i).name);
        if (isempty(core))
            continue
        end
        %
        fname = fullfile(d(i).folder, d(i).name);
        %fprintf('%s\n',fname);
        A = getImg(fname);
        A.cinfo = join(core,ctrl);
        getFlux(C,A);
        %
        if (C.opt>0)
            Z.A{k} = A;
        end
        k=k+1;
    end
    %
end


function core = findCore(C,fname)
%%---------------------------------------------------
%% return the extracted core info from fname
%%
%% 2020-08-06   Alex Szalay
%%---------------------------------------------------
    a = split(fname,'_');
    tma = str2num(a{3});
    cri = replace(a{6},'Core','');
    core = C.core(C.core.TMA==tma & strcmp(C.core.Core,cri)>0,:);
end