function C = readMetadata(root1, samp)
%%----------------------------------------------
%% Read all the metadata from the csv files.
%% opt will determine whether to read the extra stuff
%%
%% Alex Szalay, Baltimore, 2018-10-25
%%---------------------------------------------
    %
    fprintf(1,'.readMetadata(%s), %s\n', samp,...
        datestr(datetime('now')) );
    %
    % figure out and save the paths
    %
    C = [];
    C.err   = 0;
    C.root  = root1;
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
        C.Q =[];
        C.P =[];
        C.P.A = readtable([C.dest,C.samp,'_annotations.csv'],'Delimiter',',');
        C.P.R = readtable([C.dest,C.samp,'_regions.csv'],'Delimiter',',');
        C.P.V = readtable([C.dest,C.samp,'_vertices.csv'],'Delimiter',',');
        C.B   = readtable([C.dest,C.samp,'_batch.csv'],'Delimiter',',');
        C.O   = readtable([C.dest,C.samp,'_overlap.csv'],'Delimiter',',');
        C.Q.T = readtable([C.dest,C.samp,'_qptiff.csv'],'Delimiter',',');
        C.C   = readtable([C.dest,C.samp,'_constants.csv'],'Delimiter',',');
        C.R   = readtable([C.dest,C.samp,'_rect.csv'],'Delimiter',',');
        C.A   = [];
        C.qimg  = imread([C.dest,C.samp,'_qptiff.jpg']);
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