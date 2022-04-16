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
    sampletracker($mpath): base ($mpath){
        $this.getmodulenames()
    }
    #
    sampletracker($mpath, $vmq): base ($mpath){
        $this.getmodulenames()
        $this.vmq = $vmq
    }
    #
    sampletracker($mpath, $vmq, [hashtable]$modules): base ($mpath){
        $this.modules = $modules
        $this.vmq = $vmq
    }
    #
    sampletracker($mpath, $vmq, [hashtable]$modules, $slideid): base ($mpath, $slideid){
        $this.getmodulenames()
        $this.vmq = $vmq
    }
    #
    [void]defmodulestatus(){
        #
        $this.modules | ForEach-Object {
            $this.deflogpaths($_)
            $this.getlogstatus($_)
        }
        #
    }
    #
    [void]preparesample($slide, $slides){
        #
        $this.ParseAPIDdef($slide.slideid, $slides)
        $this.defbase()
        $this.defmodulestatus()
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
    #
}