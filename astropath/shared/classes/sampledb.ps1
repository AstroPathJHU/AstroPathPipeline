<# -------------------------------------------
 sampledb
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build the sample db for each
 module
 -------------------------------------------#>
class sampledb : sampletracker {
    #
    [vminformqueue]$vmq
    [hashtable]$moduleobjs = @{}
    #
    [string]$vminform_nonab_keys = (@('project','slidelog','mainlog','starttime',
        'finishtime','version','status') -join '|')
    #
    [hashtable]$status_settings
    #
    sampledb(): base ('\\bki04\astropath_processing'){
        $this.sampledbinit()
    }
    sampledb($mpath) : base ($mpath) {
        $this.sampledbinit()
    }
    sampledb($mpath, $projects) : base ($mpath){
        $this.projects = $projects
        $this.sampledbinit()
    }
    #
    sampledbinit(){
        #
        $this.importaptables($this.mpath, $true)
        $this.defmodulequeues()
        $this.status_settings = @{
            rerun_reset_status = (@($this.status.ready,
                 $this.status.finish, $this.status.error) -join '|');
            update_status = (@($this.status.waiting, $this.status.error, 
                $this.status.finish, $this.status.running) -join '|')
        }
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
        $this.defmodulelogwatchers()
        $this.refreshmoduledb()
        #
    }
    <# -----------------------------------------
    defmodulequeues
    -----------------------------------------   
    read in or create the intial module tables
    as a moduleobjs hashtable
    ----------------------------------------- #>
    [void]defmodulequeues(){
        #
        $this.vmq = vminformqueue $this.mpath
        $this.getmodulenames()
        $this.modules | ForEach-Object{
            $this.moduleobjs.($_) = modulequeue $this.mpath $_
        }
        #
    }
    <# -----------------------------------------
    defmodulewatchers
    -----------------------------------------   
    run after the sampledb is built:
    write out the module tables,
    create the module table watchers, if 
    samples were updated while the sampledb
    was being built add those slides to be checked
    ----------------------------------------- #>
    [void]defmodulewatchers(){
        #
        if(!$this.newtasks){
            $this.newtasks = @()
        }
        #
        $this.vmq.createwatchersvminformqueues()
        $this.modules | ForEach-Object{
            $this.moduleobjs.($_).createwatchersqueues()
            $this.newtasks += $this.moduleobjs.($_).newtasks
            if( $this.moduleobjs.($_).newtasks){
                write-host $_
                write-host $this.moduleobjs.($_).newtasks
            }
            $this.moduleobjs.($_).newtasks = @()
        }
        #
    }
    <# -----------------------------------------
    defmodulelogwatchers
    -----------------------------------------   
    read in the logs and create new watchers. 
    if the log was updated while sampledb was
    created this should add tasks to new tasks
    ----------------------------------------- #>
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
            $this.preparesample($slide, $this.slide_data)
            $this.defmoduletables($slide.slideid)
        }
        #
        write-progress -Activity "Checking slides" -Status "100% Complete:" -Completed
        #
    }
    <# -----------------------------------------
    defmoduletables
    For each module, create or read in 
    a module table. refresh the sample status
    in the module table. Write out main
    and project level module tables.
    ------------------------------------------
    Usage: $this.defmoduletables()
    ----------------------------------------- #>
    [void]defmoduletables($slideid){
        #
        $this.modules | ForEach-Object{
            $this.refreshmoduledb($_, $slideid)
        }
        #
    }
    #
    [void]refreshmoduledb(){
        #
        $this.refreshsampledb()
        #
        $this.modules | ForEach-Object{
            $this.moduleobjs.($_).writemainqueue()
            $this.refreshmoduledb($_)
        }
        #
    }
    #
    # refresh all the module tables for a particular sample
    # runs if a logging 'newtask' is created (i.e. a finish event)
    #
    [void]refreshsampledb(){
        #
        while($this.newtasks){
            $slide, $this.newtasks = $this.newtasks
            $this.preparesample($slide)
            $this.defmoduletables($slide)
        }
        #
    }
    #
    [void]refreshsampledb($cmodule, $project){
        #
        $this.getmodulelogs($cmodule, $project)
        $this.refreshmoduledb()
        #
    }
    <# -----------------------------------------
    refreshmoduledb
    For a module, if the main table updates
    check for edited rows. If there are new
    'rerun' lines check those against the sampledb.
    update the sample status for a module if 
    needed. write out the module table. also
    write a project level module table. 
    ------------------------------------------
    Usage: $this.refreshmoduledb(cmodule)
    ----------------------------------------- #>
    [void]refreshmoduledb($cmodule){
        #
        $this.moduleobjs.($cmodule).openmainqueue($false)
        while($this.moduleobjs.($cmodule).newtasks){
            $slide, $this.moduleobjs.($cmodule).newtasks =
             $this.moduleobjs.($cmodule).newtasks
            $this.preparesample($slide)
            $this.refreshmoduledb($cmodule, $slide)
        }
        #
        $this.moduleobjs.($cmodule).writemainqueue()
        $this.moduleobjs.($cmodule).coalescequeues()
        #
        if ($this.moduleobjs.($cmodule).newtasks){
            $this.refreshmoduledb($cmodule)
        }
        #
    }
    <# -----------------------------------------
    refreshmoduledb
    check the module level status for the specified
    sample against the module db. update the status
    for module if needed and write out main
    and project level tables.
    ------------------------------------------
    Usage: $this.refreshmoduledb(cmodule, slideid)
    ----------------------------------------- #>
    [void]refreshmoduledb($cmodule, $slideid){
        #
        if ($cmodule -match 'vminform'){
            $this.refreshmoduledbsub($cmodule, $slideid, $this.project)
        } else {
            $this.refreshmoduledbsub($cmodule, $slideid)
        }
        #
    }
    #
    [array]getslideantibodies($slideid){
        #
        $modulekeys = $this.moduleinfo.vminform.Keys
        $antibodies = $modulekeys -notmatch $this.vminform_nonab_keys
        return $antibodies
        #
    }
    #
    [void]refreshmoduledbsub($cmodule, $slideid){
        #
        $row = $this.moduleobjs.($cmodule).maincsv |
            Where-Object {$_.slideid -match $slideid}
        if (!$row){
            $this.createmoduleline($cmodule, $slideid)
        } else {
            $this.updatemoduleline($cmodule, $slideid)
        }
        #
    }
    #
    [void]createmoduleline($cmodule, $slideid){
        #
        $currentobj = $this.moduleinfo.($cmodule)
        #
        $row = [PSCustomObject]@{
            Project = $this.project
            Cohort = $this.cohort
            SlideID = $slideid
            Status = ($currentobj.status -replace ',', ';')
            isGood = 1
            StartTime = $currentobj.StartTime
            FinishTime = $currentobj.FinishTime
        }
        #
        if(!$this.moduleobjs.($cmodule).maincsv){
            $this.moduleobjs.($cmodule).maincsv = @()
        }
        $this.moduleobjs.($cmodule).maincsv += $row 
        #
    }
    #
    [void]updatemoduleline($cmodule, $slideid){
        #
        $row = $this.moduleobjs.($cmodule).maincsv |
            Where-Object {$_.slideid -match $slideid}
        #
        $slidestatus = $this.moduleinfo.($cmodule).status -replace ',', ';'
        $modulestatus = $row.status
        #
        if ($modulestatus -notmatch $slidestatus -and 
            $slidestatus -match $this.status.ready){
            # enqueue the slide and update module status in table
            $row.status = $slidestatus
        }
         #
        if ($modulestatus -notmatch ($slidestatus, $this.status.rerun -join '|') -and 
            $slidestatus -match $this.status_settings.update_status){
            $row.status = $slidestatus
        }
        #
        if ($modulestatus -match $this.status.rerun -and 
                $slidestatus -match $this.status_settings.rerun_reset_status){
            # enqueue the slide and update module status in table
            $row.status = $this.status.ready
            $this.moduleinfo.($cmodule).StartTime = $this.empty_time
            $this.moduleinfo.($cmodule).FinishTime = $this.empty_time
        }
        #
        if (
            ($row.StartTime -ne $this.moduleinfo.($cmodule).StartTime)
        ) {
            $row.StartTime = $this.moduleinfo.($cmodule).StartTime
        }
        #
        if (
            ($row.FinishTime -ne $this.moduleinfo.($cmodule).FinishTime)
        ) {
            $row.FinishTime = $this.moduleinfo.($cmodule).FinishTime
        }
        #
    }
    #
    [void]refreshmoduledbsub($cmodule, $slideid, $project){
        #
        $row = $this.moduleobjs.($cmodule).localqueue.($project) |
            Where-Object {$_.slideid -match $slideid}
        if ($row){
            $this.updatemoduleline($cmodule, $slideid, $project)
        } else{
            $this.createmoduleline($cmodule, $slideid, $project)
        }
        #
    }
    #
    [void]createmoduleline($cmodule, $slideid, $project){
        #
        if(!$this.moduleobjs.($cmodule).localqueue.($project)){
            $this.moduleobjs.($cmodule).localqueue.($project) = @()
        }
        #
        $antibodies = $this.getslideantibodies($slideid)
        #
        $row = [PSCustomObject]@{
            Project = $this.project
            Cohort = $this.cohort
            SlideID = $slideid
        }
        #
        $antibodies | ForEach-Object {
            $statusname = ($_ + '_Status')
            $startname = ($_ + '_StartTime') 
            $finishname = ($_ + '_FinishTime')
            $status = $this.moduleinfo.($cmodule).($_).status -replace ',', ';'
            $row | Add-Member -NotePropertyMembers @{
                $statusname =  $status
                $startname =  $this.moduleinfo.($cmodule).($_).StartTime
                $finishname =  $this.moduleinfo.($cmodule).($_).FinishTime
            } -PassThru
        }
        #
        $this.moduleobjs.($cmodule).localqueue.($project) += $row
        #
    }
    #
    [void]updatemoduleline($cmodule, $slideid, $project){
        #
        $antibodies = $this.getslideantibodies($slideid)
        $row = $this.moduleobjs.($cmodule).localqueue.($project) |
                Where-Object {$_.slideid -match $slideid}
        #
        $antibodies | foreach-object{
            #
            $statlabel = ($_ + '_Status')
            $startlabel = ($_ + '_StartTime')
            $finishlabel = ($_ + '_FinishTime')
            #
            $slidestatus = $this.moduleinfo.($cmodule).($_).status `
                -replace ',', ';'
            $modulestatus = $row.($statlabel)
            #
            if ($modulestatus -notmatch $slidestatus -and 
                $slidestatus -match $this.status.ready){
                # enqueue the slide 
                $row.($statlabel)= $slidestatus
            }
            #
            if ($modulestatus -notmatch ($slidestatus, $this.status.rerun -join '|') -and 
                    $slidestatus -match $this.status_settings.update_status){
                $row.($statlabel) = $slidestatus
            }
            #
            if ($modulestatus -match $this.status.rerun -and 
                $slidestatus -match $this.status_settings.rerun_reset_status){
                # enqueue the slide 
                $row.($statlabel) = $this.status.ready
                $this.moduleinfo.($cmodule).($_).StartTime = $this.empty_time
                $this.moduleinfo.($cmodule).($_).FinishTime = $this.empty_time
            }
            #
            if (
                $row.($startlabel) -ne 
                    $this.moduleinfo.($cmodule).($_).starttime
            ) {
                $row.($startlabel) = 
                    $this.moduleinfo.($cmodule).($_).starttime
            }
            #
            if (
                $row.($finishlabel) -ne 
                    $this.moduleinfo.($cmodule).($_).finishtime
            ) {
                $row.($finishlabel) = 
                    $this.moduleinfo.($cmodule).($_).finishtime
            }
        }
         #
    }
    #
    [void]updatetables($filetype){
        $this.vmq.($filetype) = $this.($filetype)
        #$this.($filetype) = $this.($filetype)
        $this.modules | ForEach-Object {
            $this.moduleobjs.($_).($filetype) = $this.($filetype)
        }
    }
    <# -----------------------------------------
    handleAPevent
    For a sample change the moduledb
    status to its sampledb status 
    ------------------------------------------
    Usage: $this.handleAPevent(cmodule, slideid)
    ----------------------------------------- #>
    #
    [void]handleAPevent($fullfile){
        #
        $fpath = Split-Path $fullfile
        $file = Split-Path $fullfile -Leaf
        #
        switch -regex ($file){
            $this.cohorts_file {
                $this.importcohortsinfo($this.mpath, $false)
                $this.updatetables('full_project_dat')
            }
            $this.paths_file {
                $this.importcohortsinfo($this.mpath, $false)
                $this.updatetables('full_project_dat')
            }
            $this.config_file {
                $this.ImportConfigInfo($this.mpath, $false)
                $this.updatetables('config_data')
            }
            $this.slide_file {
                $this.ImportSlideIDs($this.mpath, $false)
                $this.updatetables('slide_data')
            }
            $this.ffmodels_file {
                $this.ImportFlatfieldModels($this.mpath, $false)
                $this.updatetables('ffmodels_data')
            }
            $this.corrmodels_file {
                $this.ImportCorrectionModels($this.mpath, $false)
                $this.updatetables('corrmodels_data')
            }
            $this.micomp_file {
                $this.ImportMICOMP($this.mpath, $false)
                $this.updatetables('micomp_data')
            }
            $this.worker_file {$this.Importworkerlist($this.mpath, $false)}
            $this.vmq.mainqueuelocation() {
                $this.vmq.coalescevminformqueues()
                $this.refreshmoduledb('vminform')
            }
        }
        #       
        $this.projects | foreach-object{
            switch -regex ($fullfile){
                $this.vmq.localqueue.($_) {
                    $this.vmq.coalescevminformqueues($_)
                    $this.refreshmoduledb('vminform', $_)
                }
            }
        }
        #
        $this.modules | ForEach-Object{
            #
            switch -regex ($fullfile){
                $this.moduleobjs.($_).mainqueuelocation() {$this.refreshmoduledb($_)}
            }
            #
            foreach ($project in $this.projects){
                switch -regex ($fullfile){
                    $this.defprojectlogpath($_, $project) {$this.refreshsampledb($_, $project)}
                }
            }
            #
        }
        #
    }
    #
}