function T = testAll(root,varargin)
%%--------------------------------------------------
%% Run the prepSample code on all samples
%% root1: directory containg all the samples and sampldef.csv
%%   The sampledef file must have at least the following two named columns
%%      SampleID: an integer (at this stage it is not propagated)
%%      SlideID: the name of the subdirectory containing the slide data
%%      ... can have others
%% varargin: an optional range in the sequence numbers in the
%%      sampledef.csv file. No value means process all samples.
%% Returns the directory struct of the samples
%% Example:
%%      d = prepSampleAll('\\bki04\Clinical_Specimen_2');
%%      d = prepSampleAll('\\bki04\Clinical_Specimen_2',(11:20));
%% -------------------------------------------------
%% 2019-02-13   Alex Szalay, Baltimore 
%% 2020-04-22   Alex Szalay, added sampledef as an argument, 
%%                  plus do some error checking
%% 2020-06-08   Alex Szalay, changed location of samnpledef.csv file
%%                  to be in the root directory
%%--------------------------------------------------
global logctrl
    logctrl=0;
    %---------------------------------------------------------
    % set the basic params, log path and top level logfile 
    %---------------------------------------------------------
    Z = getConfig(root,'','test');
    %----------------------
    % get sampledef file
    %----------------------
    Z.T = getSampledef(Z);
    %--------------
    % get option
    %--------------
    Z.opt = [];
    if (numel(varargin)>0)
        Z.opt = varargin{1};
    end
    %------------------------------
    % set the range of samples
    %------------------------------
    if (isempty(Z.opt))
        Z.range = (1:numel(Z.T.SampleID));
    else
        Z.range = Z.opt;
    end
    %------------------------------------------------------------------
    % delete detailed logs, if in write mode, and the opt not a range
    %------------------------------------------------------------------
    if (logctrl>0 & exist(Z.logtop)==2 & isempty(Z.opt))
        delete(Z.logtop);
    end   
    %-----------------------------------------
    % loop through the samples with osGood==1
    %-----------------------------------------
    t = [];
    for i=Z.range
        if (Z.T.isGood(i)==1)
            samp = Z.T.SlideID{i};
            try
                O = test(Z.root,samp);
                t = [t;O];
            catch
            end
        end
    end
    T = cell2table(t);
    T.Properties.VariableNames={'samp',...
        'px00','px10','px01','py00','py10','py01'};
    %----------------------
    % disable hard logging
    %----------------------
    logctrl=0;
    %
end


