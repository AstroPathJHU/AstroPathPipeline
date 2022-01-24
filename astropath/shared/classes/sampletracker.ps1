<# -------------------------------------------
 sampletracker
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build the sample trackers for each
 sample and module
 -------------------------------------------#>
class sampletracker : dependencies {
    #
    sampletracker($mpath, $slideid): base ($mpath, $slideid){}
    #
    # sampletracker($mpath, $module, $batchid, $project) : base ($mpath, $module, $batchid, $project){}
    #
    [void]defmodulestatus(){
        #
        $this.getmodulenames()
        $this.modules | ForEach-Object {
            $this.deflogpaths($_)
            # create file watcher
            $this.getlogstatus($_)
        }
        #
    }
    #
    [void]defsamplelogwatcher($cmodule){
        #
        $file = $this.modulelogs.($cmodule).slidelog
        $fname = $file.Split('\\')[-1]
        $fpath = $file.replace(('\'+$fname), '')
        #
        $newwatcher = [System.IO.FileSystemWatcher]::new($fpath)
        $newwatcher.Filter = $fname
        $newwatcher.NotifyFilter = 'LastWrite'
        #
        $smpletracker = $this
        $sb = {
            $this.getlogstatus($cmodule)
        }.GetNewClosure()
        #
        $onChanged = Register-ObjectEvent $newwatcher `
            -EventName Changed `
            -SourceIdentifier ($fpath + '\' + $fname) `
            -Action $sb
    }
    #
    [void]removewatchers(){
        $this.modules | ForEach-Object {
            $SI = $this.modulelogs.($cmodule).slidelog
            $this.UnregisterEvent($SI)
        }
    }
}