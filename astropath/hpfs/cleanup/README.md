# 5.13. Clean Up
## 5.13.1. Description
Here we check that there is the proper number of orginial *im3

    [im3_err_val, expectim3num] = check_im3s(wd, sname);
    [fw_err_val] = check_fw_fw01(fw, sname, expectim3num);
    [flatw_err_val] = check_flatws(wd, sname, expectim3num);
    [xml_err_val] = check_xmls(wd, sname, expectim3num);
    [comps_err_val] = check_components(wd, sname, expectim3num);
    [tbl_err_val] = check_tables(wd, sname, expectim3num);
    
    
    cleanup_cohort(wd, fw)  
