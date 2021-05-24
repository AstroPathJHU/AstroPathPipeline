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
    csv = { 'align','batch','overlap','qptiff','constants','rect',...
            'exposures','annotations','regions','vertices',...
            'affine','fields'};
    str = { 'C.A','C.B','C.O','C.QT','C.C','C.R',...
            'C.E','C.PA','C.PR','C.PV','C.T','C.H'};
    for i=1:numel(csv)
        fname = [C.dbload,C.samp,'_',csv{i},'.csv'];
        try
            t = readtable(fname,'Delimiter',',');
            eval([str{i},'=t;']);
        catch
            msg=sprintf('WARNING: file%s not found',fname);
            logMsg(C,msg,1);
        end
    end
    %
    C.qimg = imread([C.dbload,C.samp,'_qptiff.jpg']);
    %
    % get scan for convenience from the Batch
    %
    C.scan = sprintf('Scan%d',C.B.Scan);
    %
    % get data from the Constants and insert name and values into C
    %
    for i=1:numel(C.C.value)
        name = C.C.name{i};
        val  = C.C.value(i);
        v    = num2str(val,10);
        cmd =['C.' name '=' v ';'];
        eval(cmd);
    end
    %
end

