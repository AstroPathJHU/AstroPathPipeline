function A=getOneImage(C, ncore, batchid)
    thiscore = C.core(C.core.ncore==ncore,:);
    thisctrl = C.ctrl(C.ctrl.TMA==thiscore.TMA(1) & C.ctrl.BatchID==batchid,:);
    C.samp = thisctrl.SlideID{1};
    f = fullfile(C.root,C.samp,'inform_data','Component_Tiffs','*.tif');
    d = dir(f);
    for i=1:numel(d)
        %------------------------------------
        % test if the core is on our list
        %------------------------------------
        core = findCore(C,d(i).name);
        fprintf('%d\n',core.ncore);
        if (core.ncore~=thiscore.ncore)
            continue
        end
        %
        fname = fullfile(d(i).folder, d(i).name);
        %fprintf('%s\n',fname);
        A = getImg(fname);
        %A.cinfo = join(core,ctrl);
        return
    end
    %
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
