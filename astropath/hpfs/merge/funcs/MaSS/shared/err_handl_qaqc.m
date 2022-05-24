
%% function: err_handl
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 07/07/2020
%% --------------------------------------------------------------
%% Description
% if there is an err_val that is not 0; output the corresponding message
% and return and exit code 1 to exit MaSS (errors) 0 to keep going
% (warnings)
%% --------------------------------------------------------
%%
function e_code = err_handl_qaqc(wd, sname, logstring, Markers, err_val) %#ok<INUSL>
%
if err_val ~= 0
    %
    switch err_val
        case 1
            err_str = ['MergeConfig file not ',...
                'found or could not be opened'];
            e_code = 1;
        case 2
            err_str = ['MergeConfig file read as double ',...
                'and two rows read as NaN. Changing first row to DAPI and ',...
                'ignoring the second'];
            e_code = 0;
        case 3
            err_str = ['MergeConfig file "Opal" column could ',...
                'not be parsed'];
            e_code = 1;
        case 4
            err_str = ['MergeConfig file "Coexpression" column could ',...
                'not be parsed'];
            e_code = 1;
        case 5
            err_str = ['could not find DAPI row in the ',...
                'MergeConfig file'];
            e_code = 1;
        case 6
            err_str = ['MergeConfig file should only contain ',...
                'one tumor designation'];
            e_code = 1;
        case 7
            err_str = ['MaSS algorithm can only handle ',...
                'expression markers with multiple segmentation ',...
                'algorithms; check MergeCongif file'];
            e_code = 1;
        case 8
            err_str = ['could not parse MergeConfig files'];
            e_code = 1;
        case 9
            err_str = ['could not delete previous results folders.',...
                ' Please check folder permissions and that all ',...
                'files are closed'];
            e_code = 1;
        case 10
            err_str = ['could not start par pool'];
            e_code = 1;
        case 11
            err_str = ['could not find or open tmp_forfigurestables'];
            e_code = 1;
        case 12
            err_str = ['cell seg summary.txt file ',...
                        'does not exist or is corrupt'];
            e_code = 1;
        case 13
            err_str = 'check binary segmentation maps contain all 4 tissue+cell layers';
            e_code = 1;
        case 14
            err_str = ['Could not export all QA QC images, check inForm tables'];
            e_code = 0;
        case 15
            err_str = 'check inForm output files';
            e_code = 0;
    end
    %
    %disp([sname, ';',err_str,';'])
    mywritetolog(wd, sname, logstring, err_str, 2, 'QA_QC');
    %
else
    e_code = 0;
end
end
