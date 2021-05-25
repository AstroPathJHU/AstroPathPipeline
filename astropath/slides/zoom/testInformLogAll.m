function testInformLogAll(root,varargin)
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
    Z = getConfig(root,'','testinformlog',opt);
    %-----------------------------------
    % get sampledef file and set range
    %-----------------------------------
    S = getSampledef(Z);    
    %
    if (isempty(opt))
        Z.range = (1:numel(S.SampleID));
    else
        Z.range = opt;                
    end
    %-----------------------------------------
    % loop through the samples with isGood==1
    %-----------------------------------------    
    for i=Z.range
        if (S.isGood(i)==1)
            samp = S.SlideID{i};
            testInformLog(Z,samp);
        end
    end
    %
end



function testInformFromLog(C,samp)
    %
    f = fullfile(C.root,samp,'inform_data','Component_Tiffs','Batch.log');
    
    %
end

