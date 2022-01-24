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
    #
    sampledb($module){
        $this.mpath = '\\bki04\astropath_processing'
        $this.module = $module 
    }
    sampledb($mpath, $module){
        $this.mpath = $mpath
        $this.module = $module 
    }
    sampledb($mpath, $module, $project){
        $this.mpath = $mpath
        $this.module = $module 
        $this.project = $project
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
        #
        $sampledb = $this.defsampleStages($slides)
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
    [array]defsampleStages($slides){
        #
        $slidesnotcomplete = @()
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
            $this.modules | ForEach-Object{
                #
                $loggers = [mylogger]::new($this.mpath, $this.module, $slide.slideid)
                #
                if ($this.module -match 'batch'){
                    $log.slidelog = $log.mainlog
                }
                #
                if ($this.checklog($log, $false)){
                    #
                    if (($this.('check'+$this.module)($log, $false) -eq 2)) {
                        $slidesnotcomplete += $slide
                    }
                    #
                }
                #
            }
        }
        #
        write-progress -Activity "Checking slides" -Status "100% Complete:" -Completed
        #
        return $slidesnotcomplete
        #
    }


    }