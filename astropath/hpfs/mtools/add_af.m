%% add_af
% add the autofluorescence row to the merge config table
%
%
function m = add_af(m, mnames)
m1 = array2table([m.Project(1), m.Cohort(1), m.BatchID(1), ...
    height(m) + 1]);
m1.Properties.VariableNames = {'Project','Cohort', 'BatchID', 'layer'};
%
m2 = array2table([{'AF'}, {'NA'}, {'NA'}, {'NA'}, {'NA'}]);
m2.Properties.VariableNames = mnames(2:6);
%
m3 =  array2table([0]);
m3.Properties.VariableNames = mnames(7);
%
m4 =  array2table({'0'});
m4.Properties.VariableNames = mnames(8);
%
m5 =  array2table([0]);
m5.Properties.VariableNames = mnames(9);
%
m6 =  array2table([{'NA'}, {'NA'}]);
m6.Properties.VariableNames = mnames(10:11);
%
m(end+1,:) = [m1, m2, m3, m4, m5, m6];
%
end