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
    sampletracker($mpath, $vmq, $slideid): base ($mpath, $slideid){
        $this.getmodulenames()
        $this.vmq = $vmq
    }
    #
    sampletracker($mpath, $modules, $vmq, $slideid): base ($mpath, $slideid){
        $this.modules = $modules
        $this.vmq = $vmq
    }
    #
    [void]defmodulestatus(){
        #
        $this.modules | ForEach-Object {
            $this.deflogpaths($_)
            #$this.FileWatcher($this.moduleinfo.($_).mainlog, $this.slideid, $_)
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
    #
    [void]handleAPevent($file){
        #
        $fpath = Split-Path $file
        $file = Split-Path $file -Leaf
        #
        switch -regex ($file){
            $this.cohorts_file {$this.importcohortsinfo($this.mpath, $false)}
            $this.paths_file {$this.importcohortsinfo($this.mpath, $false)}
            $this.config_file {$this.ImportConfigInfo($this.mpath, $false)}
            $this.slide_file {$this.ImportSlideIDs($this.mpath, $false)}
            $this.ffmodels_file {$this.ImportFlatfieldModels($this.mpath, $false)}
            $this.corrmodels_file {$this.ImportCorrectionModels($this.mpath, $false)}
            $this.micomp_file {$this.ImportMICOMP($this.mpath, $false)}
            $this.worker_file {$this.Importworkerlist($this.mpath, $false)}
            $this.vmq.mainqueue_file {$this.vmq.openmainvminformqueue($false)}
            $this.vmq.localqueue_file {}
        }
    }
}