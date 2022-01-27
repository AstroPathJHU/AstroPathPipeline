<# -------------------------------------------
 sampledb
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build the sample db for each
 module
 -------------------------------------------#>
class sampledb : sharedtools {
    #
    [array]$projects
    [hashtable]$sampledb = @{}
    [vminformqueue]$vmq
    #
    sampledb(){
        $this.mpath = '\\bki04\astropath_processing'
        $this.vmq = [vminformqueue]::new()
    }
    sampledb($mpath){
        $this.mpath = $mpath
        $this.vmq = [vminformqueue]::new()
    }
    sampledb($mpath, $project){
        $this.mpath = $mpath
        $this.project = $project
        $this.vmq = [vminformqueue]::new()
    }
    #
    <# -----------------------------------------
     buildsampledb
     build the sample db from the dependency checks
     ------------------------------------------
     Usage: $this.buildsampledb()
    ----------------------------------------- #>
    [void]buildsampledb(){
        #
        $slides = $this.importslideids($this.mpath)
        $this.defsampleStages($slides)
        #
    }
    <# -----------------------------------------
    defsampleStages
    For each slide, check the current module 
    and the module dependencies to create a status
    for each module and file watchers for the samples
    log
    ------------------------------------------
    Usage: $this.defNotCompletedSlides(cleanedslides)
    ----------------------------------------- #>
    [void]defsampleStages($slides){
        #
        $c = 1
        $ctotal = $slides.count
        #
        foreach($slide in $slides){
            #
            $p = [math]::Round(100 * ($c / $ctotal))
            Write-Progress -Activity "Checking slides" `
                            -Status "$p% Complete:" `
                            -PercentComplete $p `
                            -CurrentOperation $slide.slideid
            $c += 1 
            #
            $this.sampledb.($slide.slideid) = [sampletracker]::new($this.mpath, $slide.slideid, $this.vmq)
            $this.sampledb.($slide.slideid).defmodulestatus()
        }
        #
        write-progress -Activity "Checking slides" -Status "100% Complete:" -Completed
        #
    }
    #
}