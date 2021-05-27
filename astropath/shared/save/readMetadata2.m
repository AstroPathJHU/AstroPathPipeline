function C = readMetadata(C)
%%----------------------------------------------
%% Read all the metadata from the csv files.
%%   C.root is the root path to the basic data
%% 	 C.lroot is the location of the flatw files
%%   C.samp is the slideid
%%
%%
%% Alex Szalay, Baltimore, 2018-10-25
%%---------------------------------------------
    %
    % figure out and save the paths
    %
    logMsg(C,mfilename);
    %
    if (exist([C.root,'\',C.samp])~=7)
        fprintf(1,'ERROR: Sample %s does not exist\n',C.samp);
        C.err = 1;
        return
    end
    %
    % get metadata from the dbload directory
    %
    C.dbload = fullfile(C.root,C.samp,'\dbload\');
    %
    csv = {'batch','overlap','qptiff','constants','rect','exposures',...
        'annotations','regions','vertices'};
    str = {'C.B','C.O','C.T','C.C','C.R','C.E','C.P.A','C.P.R','C.P.V'};
    T   = {};
    for i=1:numel(csv)
        fname = [C.dbload,C.samp,'_',csv{i},'.csv'];
        try
            T = readtable(fname,'Delimiter',',');
            eval([str{i},'=T;']);
        catch
            msg=sprintf('ERROR: file%s not found',fname);
            logMsg(C,msg,1);
            C.err = 1;
        end
    end
    %
    C.qimg = imread([C.dbload,C.samp,'_qptiff.jpg']);
    %
    % get scan from the Batch
    %
    C.scan = sprintf('Scan%d',C.B.Scan);
    %
    % get data from the Constants
    %
    C.fwidth  = getVal(C,'fwidth');
    C.fheight = getVal(C,'fheight');
    C.pscale  = getVal(C,'pscale');
    C.qpscale = getVal(C,'qpscale');
    C.apscale = getVal(C,'apscale');
    C.xposition = getVal(C,'xposition');
    C.yposition = getVal(C,'yposition');
    C.nclip   = getVal(C,'nclip');
    C.layer   = getVal(C,'layer');
    %
end


function value = getVal(C,name)
%%-------------------------------------------------------
%% extract the value of the field from the Const struct
%%
%% Alex Szalay, Baltimore, 2018-10-29
%%-------------------------------------------------------
    %
    value = table2array(C.C(strcmp(C.C.name,name),'value'));
    %
end