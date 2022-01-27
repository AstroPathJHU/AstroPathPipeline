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
        $q = New-Object System.Collections.Queue
    }
    #
    [void]filewatchersloop(){
        #
        $this.sampledb | ForEach-Object {
            $this.filewatcher($_)
        }
    }
    #
    [void]filewatcher($sample){
        $file = $sample.modulelogs.('transfer').slidelog
        $fpath = Split-Path $file
        $fname = Split-Path $file -Leaf
        #
        $sample.createdirs($fpath)
        #
        $newwatcher = [System.IO.FileSystemWatcher]::new($fpath)
        $newwatcher.Filter = $fname
        $newwatcher.NotifyFilter = 'LastWrite'
        #
        Register-ObjectEvent $newwatcher `
            -EventName Changed `
            -SourceIdentifier ($fpath + '\' + $fname) `
            -MessageData ($sample.slideid + '-transfer') | Out-NUll
    }
    #
}