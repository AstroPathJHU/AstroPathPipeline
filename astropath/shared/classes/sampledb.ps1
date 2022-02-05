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
    [hashtable]$moduledb = @{}
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
    sampledb($mpath, $projects){
        $this.mpath = $mpath
        $this.projects = $projects
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
            $sampletracker = [sampletracker]::new($this.mpath, $slide.slideid, $this.vmq)
            $sampletracker.defmodulestatus()
            $this.sampledb.($slide.slideid) = $sampletracker
        }
        #
        write-progress -Activity "Checking slides" -Status "100% Complete:" -Completed
        #
    }
    <# -----------------------------------------
    defmoduleStages
    For each module, create or read in 
    a module table. refresh the sample status
    in the module tabl. Write out main
    and project level module tables.
    ------------------------------------------
    Usage: $this.defmoduleStages()
    ----------------------------------------- #>
    <# -----------------------------------------
    refreshmoduledb
    For a module, check each sample in 
    the sampledb against the module table.
    update the sample status for a module if 
    needed. write out the module table. also
    write a project level module table. 
    ------------------------------------------
    Usage: $this.updatemoduledb(cmodule)
    ----------------------------------------- #>
    <# -----------------------------------------
    refreshmoduledb
    check the module level status for the specified
    sample against the module db. update the status
    for module if needed and write out main
    and project level tables.
    ------------------------------------------
    Usage: $this.updatemoduledb(cmodule, slideid)
    ----------------------------------------- #>
    <# -----------------------------------------
    defmoduledb
    For a module, create or read in 
    the module table.
    ------------------------------------------
    Usage: $this.defmoduledb(cmodule)
    ----------------------------------------- #>
    <# -----------------------------------------
    comparesamplemodule
    For a sample check that the moduledb
    status matches 
    ------------------------------------------
    Usage: $this.comparesamplemodule(cmodule, slideid)
    ----------------------------------------- #>
    <# -----------------------------------------
    updatemodouledb
    For a sample change the moduledb
    status to its sampledb status 
    ------------------------------------------
    Usage: $this.updatemodouledb(cmodule, slideid)
    ----------------------------------------- #>
    <# -----------------------------------------
    writemoduledb
    write out the main module db
    ------------------------------------------
    Usage: $this.writemoduledb()
    ----------------------------------------- #>
    <# -----------------------------------------
    writemoduledb
    write out the project module db
    ------------------------------------------
    Usage: $this.writemoduledb(project)
    ----------------------------------------- #>
}