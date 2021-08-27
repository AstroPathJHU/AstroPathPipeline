%wd = '\\bki07\Clinical_Specimen_11\Batch';
%ids = {'15','16','17','18','19','20','21','22','23','24','25'};
wd = '\\bki04\Clinical_Specimen_2\Batch';
ids = {'01','02','03','04','05','06','07','08','09'};
opal_dil = {'1to10','1to100','1to100','1to200','1to100','1to100','1to50'};
ab_dil = {'NA','1to400','1to100','1to100','1to200','1to4000','1to100'};
%
for i1 = 1:length(ids)
    id = ids{i1};
    f = [wd,'\Batch_',id,'.xlsx'];
    t = readtable(f);
    t.BatchID = repmat(id, height(t),1);
    t.Opal = {'DAPI','520','540','570','620','650','690'}';
    t.OpalDilution = opal_dil';
    t.AbDilution = ab_dil';
    %
    t.Properties.VariableNames = {'BatchID','OpalLot','Opal',...
        'OpalDilution','Target','Compartment','AbClone','AbLot',...
        'AbDilution'};
    disp(t)
    writetable(t, f);
    %
    if width(t) < 11
       t2 = table();
       t2.Project = repmat(2, 1,1);
       t2.Cohort = repmat(2, 1,1);
       t2.BatchID = id;
       t = outerjoin(t2, t, 'MergeKeys',1, 'Keys',{'BatchID'});
    end
    %
    t.Properties.VariableNames = {'Project','Cohort','BatchID','OpalLot','Opal',...
        'OpalDilution','Target','Compartment','AbClone','AbLot',...
        'AbDilution'};
    disp(t)
    %writetable(t, f);
    f = replace(f,'xlsx','csv');
    writetable(t,f);
end
%
for i1 = 1:length(ids)
    id = ids{i1};
    f = [wd,'\MergeConfig_',id,'.xlsx'];
    t = readtable(f);
    %
    if height(t) < 8
        t(end + 1, :) = t(1, :);
        t(end, 2) = {'AF'};
        t(end, 3) = {'NA'};
    end
    %
    if width(t) > 10
        t.Var11 = [];
    end
    t.BatchID = repmat(id, height(t),1);
    t.Opal = {'DAPI','520','540','570','620','650','690', 'AF'}';
    t.CoexpressionStatus = {'NA','620,690', '570','540','NA','540,570','NA','NA'}';
    disp(t)
    %delete(f)
    %writetable(t, f);
    %
    if width(t) < 12
       t2 = table();
       t2.Project = repmat(2, 1,1);
       t2.Cohort = repmat(2, 1,1);
       t2.BatchID = id;
       t = outerjoin(t2, t, 'MergeKeys',1, 'Keys',{'BatchID'});
    end
    t.Properties.VariableNames = {'Project','Cohort','BatchID','Opal',...
        'Target','Compartment','TargetType','CoexpressionStatus',...
        'SegmentationStatus','SegmentationHierarchy',...
        'NumberofSegmentations','ImageQA'};
    t.layer = [1:8]';
    t = t(:, {'Project','Cohort','BatchID', 'layer', 'Opal',...
        'Target','Compartment','TargetType','CoexpressionStatus',...
        'SegmentationStatus','SegmentationHierarchy',...
        'NumberofSegmentations','ImageQA'});
    disp(t)
    f = replace(f,'xlsx','csv');
    writetable(t,f);
end