function C = readMetadata(root1,samp)
%%----------------------------------------------
%% Read all the metadata from the csv files.
%% opt will determine whether to read the extra stuff
%%
%% Alex Szalay, Baltimore, 2018-10-25
%%---------------------------------------------
    %
    logMsg(samp,'readMetadata');
    %
    % figure out and save the paths
    %
    %root1 = '\\bki02\e\Clinical_Specimen';
    root2 = '\\bki02\f\zoomcube';
    %
    C = [];
    C.err   = 0;
    C.root  = root1;
    C.root2 = root2;
    C.samp  = samp;
    C.dest  = [C.root '\' C.samp '\dbload\'];
    %
    if (exist([C.root,'\',C.samp])~=7)
        fprintf(1,'ERROR: Sample %s does not exist\n',C.samp);
        C.err = 1;
        return
    end
    %
    % get metadata from the dbload directory
    %
    try 
        %C.Q =[];
        %C.P =[];
        %C.P.A = readtable([C.dest,C.samp,'_annotations.csv'],'Delimiter',',');
        %C.P.R = readtable([C.dest,C.samp,'_regions.csv'],'Delimiter',',');
        %C.P.V = readtable([C.dest,C.samp,'_vertices.csv'],'Delimiter',',');
        C.B   = getTable(C,'batch');
        %C.O   = readtable([C.dest,C.samp,'_overlap.csv'],'Delimiter',',');
        %C.Q.T = readtable([C.dest,C.samp,'_qptiff.csv'],'Delimiter',',');
        %C.Q.img  = imread([C.dest,C.samp,'_qptiff.jpg']);
        C.C   = getTable(C,'constants');
        C.R   = getTable(C,'fields');
        %C.R   = getTable(C,'rect');
        %C.S   = readtable([C.dest,C.samp,'_shift.csv'],'Delimiter',',');
        %C.S   = readtable([C.dest,C.samp,'_align.csv'],'Delimiter',',');
        %C.A   = [];
    catch
        fprintf(1,'ERROR in reading metadata files in %s\n',C.dest);
        C.err = 1;
        return
    end
    %
    % get data from the Batch
    %
    C.scan = sprintf('Scan%d',C.B.Scan);
    %
    % get data from the Constants
    %
    C.fwidth  = getVal(C,'fwidth');
    C.fheight = getVal(C,'fheight');
    C.pscale  = getVal(C,'pscale');
    C.qpscale = getVal(C,'qpscale');
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


function t = getTable(C, name)
%%--------------------------------------------------------------
%% Read a table from a CSV file in the sample\dbload directory
%%
%%--------------------------------------------------------------
    %
    logMsg(C.samp,['reading _' name '.csv']);
    %
    f = [C.dest,C.samp,'_' name '.csv'];
    %
    t = [];
    if (exist(f)==0)
        logMsg(C.samp,sprintf('_%s.csv file not found',name));     
        return
    end
    %
    % read the file
    % 
    try
        t = readtable(f,'Delimiter',',');
    catch
        logMsg(C.samp,sprintf('Could not read _%s.csv file',name));
        C.err = 1;
    end           
    %
end
