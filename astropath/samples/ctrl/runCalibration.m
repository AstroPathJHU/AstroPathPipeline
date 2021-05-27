function C=runCalibration(project,varargin)
%%-------------------------------------
%% run the whole calibration process
%%
%% 2020-11-02   Alex Szalay
%%-------------------------------------
global logctrl
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    if (opt==0)
        logctrl=1;
    end
    %
    root = getRoot(project);
    samp = '';    
    C = getConfig(root,'','ctrl');
    C.opt=opt;
    %-------------------------------------------
    % set the directories for input and output
    % and other parameters
    %-------------------------------------------
    C.tiff   = '\inform_data\Component_TIFFs';
    C.batch  = fullfile(C.root,'Batch');
    C.dbload = fullfile(C.root,'Ctrl');
    %
    C.height = 3008;
    C.width  = 4028;
    %----------------------
    % read the directories
    %----------------------
    C.d = dir([C.root,'\Control_TMA*']);
    if (numel(C.d)==0)
        C.err=1;
        logMsg(C,'No Control samples file found',1);
        logctrl=0;
        return
    end
    %
    if (opt==-1)
        logctrl=0;
        return
    end    
    %
    C.core = getCoreInfo(C);
    if (C.err==1)
        logctrl=0;
        return
    end
    %
    C.ctrl = getCtrlInfo(C);
    %
    C.B    = getBatchInfo(C);
    if (C.err==1)
        logctrl=0;
        return
    end
    %
    if (opt==-2)
        logctrl=0;
        return
    end
    %
    if (C.opt==0)
        C.fout = fullfile(C.dbload,...
            sprintf('project%d_ctrlfluxes.csv',C.project));
        if (exist(C.fout)>0)
            delete(C.fout);
        end
        dlmwrite(C.fout,'project,cohort,core,tma,batch,marker,m1,m2',...
            'delimiter','','newline','pc');       
    end
    %
    for n=1:numel(C.ctrl.Ctrl)
        C.samp = 'Ctrl';
        getCoreData(C,n);
    end
    %------------------------------------
    % collect the csv files for loading,
    % write project<prno>_loadfiles.csv
    %------------------------------------
    %C = scanCsv(C);
    %
    logMsg(C,'runCalibration finished');
    %
end