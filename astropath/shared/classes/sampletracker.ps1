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
    [vminformqueue]$vmq
    #
    sampletracker($mpath, $slideid, $vmq): base ($mpath, $slideid){
        $this.getmodulenames()
        $this.vmq = $vmq
    }
    #
    sampletracker($mpath, $modules, $slideid, $vmq): base ($mpath, $slideid){
        $this.modules = $modules
        $this.vmq = $vmq
    }
    #
    # sampletracker($mpath, $module, $batchid, $project) : base ($mpath, $module, $batchid, $project){}
    #
    [void]defmodulestatus(){
        #
        $this.modules | ForEach-Object {
            $this.deflogpaths($_)
            #$this.FileWatcher($this.moduleinfo.($_).slidelog)
            $this.getlogstatus($_)
        }
        #
    }
    #
    [void]removewatchers(){
        if ($this.modules){
            $this.modules | ForEach-Object {
                $SI = $this.moduleinfo.($_).slidelog
                $this.UnregisterEvent($SI)
            }
        }
    }
}