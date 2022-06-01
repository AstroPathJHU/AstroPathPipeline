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
    [switch]$wfon = $false
    #
    sampletracker($mpath) : base ($mpath){
        #
        if ($this.wfon){
            $this.writeoutput("Starting the AstroPath Pipeline")
            $this.writeoutput(" Importing AstroPath tables from: " + $this.mpath)
        }
        $this.importaptables($this.mpath, $false)
        $this.getmodulestatus()
        #
        if ($this.wfon){
            $this.writeoutput(" AstroPath Modules: " + $this.modules)
            $this.writeoutput(" Importing AstroPath logs")
        }
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
    sampletrackerinit(){
        
    }
    #
    [void]defmodulestatus($c, $ctotal){
        #
        $this.modules | & { process {
            $this.progressbar($c, $ctotal, ($this.slideid, "update [$_]" -join ' - ')) 
            $this.deflogpaths($_)
            $this.getlogstatus($_)
        }}
        #
    }
    #
    [void]defmodulestatus(){
        #
        $this.modules | & { process { 
            $this.deflogpaths($_)
            $this.getlogstatus($_)
        }}
        #
    }
    #
    [void]preparesample($slide, $c, $ctotal){
        #
        $this.importslideids($this.mpath)
        $this.ParseAPIDdef($slide)
        $this.defbase()
        $this.moduleinfo.project = $this.project
        $this.getantibodies()
        $this.defmodulestatus($c, $ctotal)
        #
    }
    #
    [void]preparesample($slide, $slides){
        #
        $this.ParseAPIDdef($slide.slideid, $slides)
        $this.defbase()
        $this.moduleinfo.project = $this.project
        $this.getantibodies()
        $this.defmodulestatus()
        #
    }
    #
    [void]preparesample($slide){
        #
        $this.importslideids($this.mpath)
        $this.ParseAPIDdef($slide)
        $this.defbase()
        $this.moduleinfo.project = $this.project
        $this.getantibodies()
        $this.defmodulestatus()
        #
    }
    #
    [void]preparesample(){
        #
        $this.defbase()
        $this.moduleinfo.project = $this.project
        $this.getantibodies()
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