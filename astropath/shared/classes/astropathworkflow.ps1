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
    [void]launchwfnobuild(){
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
    [void]visLOCALQUEUE($cmodule, $cproject){
        Write-host ($this.moduleobjs.($cmodule).localqueue.($cproject) |
            Format-Table |
            Out-String)
    }
    #
    [void]visLOCALQUEUE($cmodule, $cproject, $COLUMN){
        Write-host ($this.moduleobjs.($cmodule).localqueue.($cproject).($COLUMN) |
            Format-Table |
            Out-String)
    }
    #
    [void]visMAINQUEUE($cmodule){
        Write-host ($this.moduleobjs.($cmodule).MAINCSV |
            Format-Table |
            Out-String)
    }
    #
    [void]visMAINQUEUE($cmodule, $cproject){
        Write-host ($this.moduleobjs.($cmodule).MAINCSV |
            where-object { $_.project -contains $cproject} |
            Format-Table |
            Out-String)
    }
    #
    [void]visMAINQUEUE($cmodule, $cproject, $COLUMN){
        Write-host ($this.moduleobjs.($cmodule).MAINCSV.($COLUMN) |
            where-object { $_.project -contains $cproject} |
            Format-Table |
            Out-String)
    }
    #
    [void]visTABLE($mytable){
        Write-host ($mytable |
            Format-Table |
            Out-String)
    }
    #
 }
