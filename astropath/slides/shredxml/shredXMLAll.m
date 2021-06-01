function Z = shredXMLAll(root,varargin)
%%--------------------------------------------------
%% Run the shredXMLAll code on all samples
%% root1: directory containg all the samples
%% sampledef: name of the .csv file
%% The file must have the following column (can have others)
%%      SampleID: an integer (at this stage it is not propagated)
%%  varargin: an optional range of sampleID 
%% Example:
%%      shredXMLAll('\\bki04\Clinical_Specimen2',(100:120));
%% or   shredXMLAll('\\bki04\Clinical_Specimen2');
%% ---------------------------------------------------------------------
%% 2019-02-13   Alex Szalay, created
%% 2020-04-22   Alex Szalay, added sampledef as an argument, 
%%                  plus do some error checking
%% 2020-06-07   Alex Szalay, now placed sampledef.csv file in the root1
%%----------------------------------------------------------------------
global logctrl
    logctrl=1;
    %--------------
    % get option
    %--------------
    opt = [];
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %---------------------------------------------------------
    % set the basic params, log path and top level logfile 
    %---------------------------------------------------------
%     if (isempty(opt))
%         Z = getConfig(root,'','shredxml');
%     else
%         Z = getConfig(root,'','shredxml',opt);
%     end
    Z = getConfig(root,'','shredxml',opt);
    if (Z.err==1)
        return
    end
    %-----------------------------------
    % get sampledef file and set range
    %-----------------------------------
    S = getSampledef(Z);
    if (Z.err==1)
        return
    end
    %
    if (isempty(opt))
        Z.range = (1:numel(S.SampleID));
    else
        Z.range = opt;                
    end
    %------------------------------------------------------------------
    % delete detailed logs, if in write mode, and the opt not a range
    %------------------------------------------------------------------
    %
    if (logctrl>0 & exist(Z.logtop)==2 & isempty(Z.opt))
        delete(Z.logtop);
    end
    %-----------------------------------------
    % loop through the samples with osGood==1
    %-----------------------------------------    
    for i=Z.range
        if (S.isGood(i)==1)
            samp = S.SlideID{i};
            shredXML(root,samp);
        end
    end
    %
end


        

