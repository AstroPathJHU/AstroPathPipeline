﻿<# -------------------------------------------
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
    [System.Collections.Concurrent.ConcurrentDictionary[string,object]]$sampledb = @{}
    [System.Collections.Concurrent.ConcurrentDictionary[string,object]]$moduledb = @{}
    [vminformqueue]$vmq
    [hashtable]$moduleobjs
    [array]$newfinishedtasks
    #
    sampledb(){
        $this.sampledbinit('\\bki04\astropath_processing')
    }
    sampledb($mpath){
        $this.sampledbinit($mpath)
    }
    sampledb($mpath, $projects){
        $this.projects = $projects
        $this.sampledbinit($mpath)
    }
    #
    sampledbinit($mpath){
        #
        $this.mpath = $mpath
        $this.importaptables($this.mpath, $true)
        $this.defmodulequeues()
        $this.getmodulelogs()
        #
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
        $this.defsampleStages()
        $this.defmodulewatchers()
        $this.getmodulelogwatchers()
        #
    }
    #
    [void]defmodulequeues(){
        #
        $this.vmq = vminformqueue $this.mpath
        $this.getmodulenames()
        $this.modules | ForEach-Object{
            $this.moduleobjs.($_) = modulequeue $this.mpath $_
        }
        #
    }
    #
    [void]defmodulewatchers(){
        #
        $this.vmq.createwatchersvminformqueues()
        $this.modules | ForEach-Object{
            $this.moduleobjs.($_).createwatchersqueues()
        }
        #
    }
    #
    [void]defmodulelogwatchers(){
        #
        $this.getmodulelogs($true)
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
    [void]defsampleStages(){
        #
        $c = 1
        $ctotal = $this.slide_data.count
        $sampletracker = sampletracker -mpath $this.mpath -vmq $this.vmq -modules $this.modules 
        #
        foreach($slide in $this.slide_data){
            #
            $p = [math]::Round(100 * ($c / $ctotal))
            Write-Progress -Activity "Checking slides" `
                            -Status ("$p% Complete: Slide " +  $slide.slideid)`
                            -PercentComplete $p `
                            -CurrentOperation $slide.slideid
            $c += 1 
            #
            $sampletracker.preparesample($slide, $this.slide_data)
            $this.sampledb.($slide.slideid) = $sampletracker.moduleinfo
        }
        #
        write-progress -Activity "Checking slides" -Status "100% Complete:" -Completed
        #
    }
    <# -----------------------------------------
    defsampleStagesParallel
    For each slide, check the current module 
    and the module dependencies to create a status
    for each module and file watchers for the samples
    log
    adopted from: https://stackoverflow.com/questions/67570734/powershell-foreach-object-parallel-how-to-change-the-value-of-a-variable-outsid
    ------------------------------------------
    Usage: $this.defNotCompletedSlides(cleanedslides)
    ----------------------------------------- #>
    [void]defsampleStagesParallel($slides){
        #
        $queue = [System.Collections.Queue]::new()
        1..$slides.Count | ForEach-Object { $queue.Enqueue($_) }
        $syncQueue = [System.Collections.Queue]::synchronized($queue)
        $sampletracker = [sampletracker]::new($this.mpath)
        #
        $parpool = $slides | ForEach-Object -AsJob -ThrottleLimit 6 -Parallel {
            $sdbcopy = $using:this.sampledb
            $sqcopy = $using:syncQueue
            # might need to import module here??
            #
            $sampletracker.ParseAPIDdef($_.slideid, $slides)
            $sampletracker.defbase()
            $sampletracker.defmodulestatus()
            $sdbcopy.($_.slideid) = $sampletracker.moduleinfo
            #
            $sqCopy.Dequeue()
        }
        #
        while ($parpool.State -eq 'Running') {
            if ($syncQueue.Count -gt 0) {
                $p = ((1 / $syncQueue.Count) * 100)
                Write-Progress -Activity "Checking slides" `
                    -Status "$p% Complete:" `
                    -PercentComplete $p
                Start-Sleep -Milliseconds 100
            }
        }
        #
        write-progress -Activity "Checking slides" -Status "100% Complete:" -Completed
        #
    }
    <#-----------------------------------------
    open
    ----------------------------------------- #>
    <# -----------------------------------------
    defmoduleStages
    For each module, create or read in 
    a module table. refresh the sample status
    in the module table. Write out main
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
    #
    [void]getmodulelogs(){
        $this.getmodulelogs($false)
    }
    #
    [void]getmodulelogs($createwatcher){
        #
        $this.getmodulenames()
        foreach ($module in $this.modules){
            $projects = $this.getapprojects()
            $projects | foreach-object{
                #
                if($this.modulelogs.($module).($_)){
                    $oldlog = $this.modulelogs.($module).($_)
                } else {
                    $oldlog = @()
                }
                #
                $this.modulelogs.($module).($_) = 
                    $this.importlogfile($module, $project, $createwatcher)
                #
                $this.getnewloglines($oldlog, $this.modulelogs.($module).($_))
                #
            }
        }
        #
    }
    #
    [void]getnewloglines($oldlog, $newlog){
        #  
        if (!$this.newfinishedtasks){
            $this.newfinishedtasks = @()
        }
        #
        if ($oldlog){
            $newlog_finishlines = $newlog |
                Where-Object {$_.Message -match '^FINISH'}
            #
            $newlog_rows = @()
            foreach ($line in $newlog_finishlines){
                $newlog_rows += $line -join ';'
            }
            #
            $oldlog_rows = @()
            foreach ($line in $oldlog){
                $oldlog_rows += $line -join ';'
            }
            #
            $cmp = compare-object $newlog_rows $oldlog_rows -Property 'SlideID','Date' |
                 Where-Object {$_.SideIndicator -eq '<='}
            $this.newfinishedtasks += $cmp.SlideID
        } 
        #
    }
    #
    [void]handleAPevent($fullfile){
        #
        $fpath = Split-Path $fullfile
        $file = Split-Path $fullfile -Leaf
        #
        switch -regex ($file){
            $this.cohorts_file {$this.importcohortsinfo($this.mpath, $false)}
            $this.paths_file {$this.importcohortsinfo($this.mpath, $false)}
            $this.config_file {
                $this.ImportConfigInfo($this.mpath, $false)
                $this.vmq.config_data = $this.config_data
            }
            $this.slide_file {$this.ImportSlideIDs($this.mpath, $false)}
            $this.ffmodels_file {$this.ImportFlatfieldModels($this.mpath, $false)}
            $this.corrmodels_file {$this.ImportCorrectionModels($this.mpath, $false)}
            $this.micomp_file {$this.ImportMICOMP($this.mpath, $false)}
            $this.worker_file {$this.Importworkerlist($this.mpath, $false)}
            $this.vmq.mainqueuelocation() {$this.coalescevminformqueues()}
        }
        #       
        $this.projects | foreach-object{
            switch -regex ($fullfile){
                $this.vmq.localqueue.($_) {$vmq.coalescevminformqueues($_)}
            }
        }
        #
    }
    #
}