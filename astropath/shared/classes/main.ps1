<# -------------------------------------------
 main
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build the sample trackers for each
 sample and module
 -------------------------------------------#>
 class main {
    #
    [hashtable]$sampledb
    #
    main(){
        $this.go($this.mpath)
    }
    #
    main($mpath){
        $this.go($mpath)
    }
    #
    [void]go($mpath){   
        $newsampledb = [sampledb]::new($mpath)
        $newsampledb.buildsampledb()
        $this.sampledb = $newsampledb.sampledb
    }
    #
}