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
    sampletracker($mpath) : base ($mpath){
        #
        $this.importaptables($this.mpath, $false)
        $this.getmodulestatus()
        $this.getmodulelogs()
        #
    }
    #
    sampletracker($mpath, $vmq): base ($mpath){
        #
        $this.importaptables($this.mpath, $false)
        $this.getmodulestatus()
        $this.getmodulelogs()
        $this.vmq = $vmq
        #
    }
    #
    sampletracker($mpath, $vmq, $modules): base ($mpath){
        #
        $this.modules = $modules
        $this.vmq = $vmq
        $this.getmodulestatus()
        $this.getmodulelogs()
        #
    }
    #
    sampletracker($mpath, $vmq, $modules, $modulelogs): base ($mpath){
        #
        $this.modules = $modules
        $this.getmodulestatus()
        $this.modulelogs = $modulelogs
        $this.vmq = $vmq
        #
    }
    #
    sampletracker($mpath, $vmq, $modules, $modulelogs, $slideid): base ($mpath, $slideid){
        #
        $this.modules = $modules
        $this.getmodulestatus()
        $this.modulelogs = $modulelogs
        $this.vmq = $vmq
        #
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
        $this.moduleinfo.project = $this.project
        $this.defmodulestatus()
        #
    }
    #
    [void]preparesample($slide){
        #
        $this.importslideids($this.mpath) | Out-Null
        $this.ParseAPIDdef($slide)
        $this.defbase()
        $this.moduleinfo.project = $this.project
        $this.defmodulestatus()
        #
    }
    #
    [void]preparesample(){
        #
        $this.defbase()
        $this.moduleinfo.project = $this.project
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