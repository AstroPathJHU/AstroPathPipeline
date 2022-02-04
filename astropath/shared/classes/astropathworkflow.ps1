<# -------------------------------------------
 astropathworkflow
 created by: Benjamin Green - JHU
 Last Edit: 01.26.2022
 --------------------------------------------
 Description
 methods used to build and excute the astropath
 workflow
 -------------------------------------------#>
 class astropathworkflow : astropathwftools {
    #
    [hashtable]$samplestatus

    #
    astropathworkflow(){}
    astropathworkflow($login) : base($login){}
    astropathworkflow($login, $mpath) : base($login, $mpath){}
    astropathworkflow($login, $mpath, $project) : base($login, $mpath, $project){}
    #
    [void]launchworkflow(){
        $this.defsampledb()
    }
    #
    [void]defsampledb(){
        $this.buildsampledb()
    }




 }
