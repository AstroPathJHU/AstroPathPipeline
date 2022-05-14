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
    [void]launchwf(){
        #
        $this.buildsampledb() 
        #
        while(1){
            $this.defworkerlist()
            $this.distributetasks()
            $this.WaitAny()
        }
    }
    #
    [void]launchwfnoloop($notasks){
        #
        $this.buildsampledb()
        $this.defworkerlist()
        #
        if ($notasks){
            $this.WaitAny()
        } else {
            $this.distributetasks()
            $this.WaitAny()
        }
        #
    }
    #
 }
