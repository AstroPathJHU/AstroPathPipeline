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
    [hashtable]$moduletaskqueue = @{}
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
        $this.writeoutput("Starting the AstroPath Pipeline")
        $this.writeoutput(" Imported AstroPath tables from: " + $this.mpath)
        $this.writeoutput(" Modules: " + $this.modules)
        #
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
        $this.newtasks = @()
        $this.defsampleStages()
        $this.writeoutput(" Creating file watchers")
        $this.defmodulewatchers()
        $this.defmodulelogwatchers()
        $this.importaptables($this.mpath, $true)
        $this.writeoutput(" Checking for updates during aggregation")
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
        $this.writeoutput(" Building module tables")
        $this.writeoutput("     Updating: [vminform] queue")
        $this.vmq = vminformqueue $this.mpath
        $this.writeoutput("     [vminform] queue updated")
        $this.getmodulenames()
        $this.modules | & { process {
            $this.writeoutput("     Updating: [$_]")
            $this.moduleobjs.($_) = modulequeue $this.mpath $_
            $this.moduletaskqueue.($_) = New-Object System.Collections.Generic.Queue[array]
        }}
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
        $this.vmq.createwatchersvminformqueues()    
        #
        $this.modules | & { process {
            $this.moduleobjs.($_).createwatchersqueues()
            $this.newtasks += $this.addnewtasks($this.moduleobjs.($_).newtasks)
            $this.moduleobjs.($_).newtasks = @()
        }}
        #
        [System.GC]::GetTotalMemory($true) | out-null
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
        $this.writeoutput(" Building sample status database")
        #
        foreach($slide in $this.slide_data){
            #
            $this.progressbar($c, $ctotal, $slide.slideid)
            $c += 1 
            $this.preparesample($slide, $this.slide_data)
            $this.defmoduletables($slide.slideid)
            #
        }
        #
        $this.progressbarfinish()
        $this.writeoutput(" Sample status database built")
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
        $this.modules | & { process {
            $this.refreshmoduledb($_, $slideid)
        }}
        #
    }
    <# -----------------------------------------
    # refresh all the module tables for a particular sample
    # runs if a logging 'newtask' is created (i.e. a finish event)
    ----------------------------------------- #>
    [void]refreshsampledb(){
        #
        if ($this.newtasks){
            $this.writeoutput(" refreshing sample data base for new tasks")
        }
        #
        $ctotal = $this.newtasks.count
        $c = 1
        #
        while($this.newtasks){
            #
            $slide, $this.newtasks = $this.newtasks
            $this.progressbar($c, $ctotal, $slide)
            $c += 1 
            $this.preparesample($slide)
            $this.defmoduletables($slide)
            #
        }
        #
    }
    #
    [void]refreshsampledb($cmodule){
        #
        $ctotal = $this.newtasks.count
        $c = 1
        #
        if ($this.newtasks){
            $this.writeoutput(" updating module database: $cmodule")
        }
        #
        while($this.newtasks){
            $slide, $this.newtasks = $this.newtasks
            $this.progressbar($c, $ctotal, $slide)
            $c += 1 
            $this.preparesample($slide)
            $this.refreshmoduledb($cmodule, $slide)
            $this.moduleobjs.($cmodule).writelocalqueue(
                $this.project)
        }
        #
        $this.refreshmoduledb($cmodule)
        #
    }
    #
    [void]refreshsampledb($cmodule, $cproject){
        #
        $this.getmodulelogs($cmodule, $cproject)
        $this.refreshmoduledb()
        #
    }
    <# -----------------------------------------
    refreshmoduledb
    check the status of 'updated' slideids in
    the newtasks variable. then check the 
    corresponding module tables and queues. 
    ------------------------------------------
    Usage: $this.refreshmoduledb()
    ----------------------------------------- #>
    #
    [void]refreshmoduledb(){
        #
        $this.refreshsampledb()
        #
        $this.modules | & { process {
            $this.moduleobjs.($_).writemoduledb()
            $this.refreshmoduledb($_)
        }}
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
        $ctotal = $this.moduleobjs.($cmodule).newtasks.count
        $c = 1
        #
        $this.moduleobjs.($cmodule).openmainqueue($false)
        if ($this.moduleobjs.($cmodule).newtasks){
            $this.writeoutput(" updating module database: $cmodule")
        }
        #
        while($this.moduleobjs.($cmodule).newtasks){
            $slide, $this.moduleobjs.($cmodule).newtasks =
             $this.moduleobjs.($cmodule).newtasks
            $this.progressbar($c, $ctotal, $slide)
            $c += 1 
            $this.preparesample($slide)
            $this.refreshmoduledb($cmodule, $slide)
        }
        #
        $this.moduleobjs.($cmodule).writemoduledb()
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
    for module if needed and enqueue the tasks if
    needed and project is in module_project_data.(cmodule)
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
    <# -----------------------------------------
    refreshmoduledbsub
    -----------------------------------------
    preforms the module table updates for the
    specified slide and moduel. overload (3) 
    to include the project for vminform task
    ----------------------------------------- #>
    [void]refreshmoduledbsub($cmodule, $slideid){
        #
        $row = $this.moduleobjs.($cmodule).maincsv | & { process {
            if ($_.slideid -contains $slideid) { $_ }}}
        if (!$row){
            $this.createmoduleline($cmodule, $slideid)
        } else {
            $this.updatemoduleline($cmodule, $slideid)
        }
        #
    }
    #
    [void]refreshmoduledbsub($cmodule, $slideid, $cproject){
        #
        $row = $this.moduleobjs.($cmodule).localqueue.($cproject) | & { process {
            if ($_.slideid -contains $slideid) { $_ }}}
        if ($row){
            $this.updatemoduleline($cmodule, $slideid, $cproject)
        } else{
            $this.createmoduleline($cmodule, $slideid, $cproject)
        }
        #
    }
    <# -----------------------------------------
    createmoduleline
    -----------------------------------------
    creates the module lines for the
    specified slide. overload (3) to include the
    project for vminform task
    ----------------------------------------- #>
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
        if ($currentobj.status -match $this.status.ready){
            $this.enqueuetask($cmodule, $slideid, $currentobj)
        }
        #
    }
    #
    [void]createmoduleline($cmodule, $slideid, $cproject){
        #
        if(!$this.moduleobjs.($cmodule).localqueue.($cproject)){
            $this.moduleobjs.($cmodule).localqueue.($cproject) = @()
        }
        #
        $cmoduleinfo = $this.moduleinfo.($cmodule)
        $this.getantibodies($cproject)
        #
        $row = [PSCustomObject]@{
            Project = $this.project
            Cohort = $this.cohort
            SlideID = $slideid
        }
        #
        $this.antibodies | & { process {
            $statusname = ($_ + '_Status')
            $startname = ($_ + '_StartTime') 
            $finishname = ($_ + '_FinishTime')
            $status = $this.moduleinfo.($cmodule).($_).status -replace ',', ';'
            if ($status -match $this.status.ready){
                $this.enqueuetask($cmodule, $slideid, $cmoduleinfo)
            }
            $row | Add-Member -NotePropertyMembers @{
                $statusname =  $status
                $startname =  $this.moduleinfo.($cmodule).($_).StartTime
                $finishname =  $this.moduleinfo.($cmodule).($_).FinishTime
            } -PassThru
        }}
        #
        $this.moduleobjs.($cmodule).localqueue.($cproject) += $row
        #
    }
    <# -----------------------------------------
    updatemoduleline    
    -----------------------------------------
    updates the preexisting module lines for the
    specified slide. overload (3) to include the
    project for vminform task
    ----------------------------------------- #>
    [void]updatemoduleline($cmodule, $slideid){
        #
        $row = $this.moduleobjs.($cmodule).maincsv | & { process {
            if ($_.slideid -contains $slideid) { $_ }}}
        #
        $statlabel = 'Status'
        $startlabel = 'StartTime'
        $finishlabel = 'FinishTime'
        #
        $cmoduleinfo = $this.moduleinfo.($cmodule)
        #
        $this.updatemodulesub($cmodule, $slideid, $row, $cmoduleinfo,
            $statlabel, $startlabel, $finishlabel)
        #
    }
    #
    [void]updatemoduleline($cmodule, $slideid, $cproject){
        #
        $this.getantibodies($cproject)
        $row = $this.moduleobjs.($cmodule).localqueue.($cproject) | & { process {
                if ($_.slideid -contains $slideid) { $_ }}}
        #
        $this.antibodies | & { process {
            #
            $statlabel = ($_ + '_Status')
            $startlabel = ($_ + '_StartTime')
            $finishlabel = ($_ + '_FinishTime')
            #
            $cmoduleinfo = $this.moduleinfo.($cmodule).($_)
            #
            $this.updatemodulesub($cmodule, $slideid, $row, $cmoduleinfo,
                $statlabel, $startlabel, $finishlabel)
            #
        }}
         #
    }
    <# -----------------------------------------
    updatemodulesub
    -----------------------------------------
    performs the specified updating actions 
    for either vminform or a normal module
    ----------------------------------------- #>
    [void]updatemodulesub($cmodule, $slideid, $row, 
        $cmoduleinfo, $statlabel,
        $startlabel, $finishlabel){
        #
        $slidestatus = $cmoduleinfo.status -replace ',', ';'
        $modulestatus = $row.($statlabel)
        #
        if ($slidestatus -match $this.status.ready){
            $this.enqueuetask($cmodule, $slideid, $cmoduleinfo)
            $row.($statlabel) = $slidestatus
        }
         #
        if ($modulestatus -notmatch (
                [regex]::Escape($slidestatus), $this.status.rerun -join '|') -and 
            $slidestatus -match $this.status_settings.update_status){
            $row.($statlabel) = $slidestatus
        }
        #
        if ($modulestatus -match $this.status.rerun -and 
            $slidestatus -match $this.status_settings.rerun_reset_status){
            $this.enqueuetask($cmodule, $slideid, $cmoduleinfo)
            $row.($statlabel) = $this.status.ready
            $cmoduleinfo.StartTime = $this.empty_time
            $cmoduleinfo.FinishTime = $this.empty_time
        }
        #
        if (
            ($row.($startlabel) -ne $cmoduleinfo.StartTime)
        ) {
            $row.($startlabel)= $cmoduleinfo.StartTime
        }
        #
        if (
            ($row.($finishlabel) -ne $cmoduleinfo.FinishTime)
        ) {
            $row.($finishlabel)= $cmoduleinfo.FinishTime
        }
        #
    }
    #
    [void]enqueuetask($cmodule, $slideid, $cmoduleinfo){
        #
        $cqueue = $this.moduletaskqueue.($cmodule)
        #
        if ($this.moduleinfo.project -and 
            $this.moduleinfo.project -notmatch
            ($this.matcharray(
                $this.module_project_data.($cmodule)
        ))){
            return
        }
        #
        switch -exact ($cmodule){
            'vminform' {
                #
                if ($cmoduleinfo.taskid -and 
                    !($cqueue -match ('^' + $cmoduleinfo.taskid))){
                    $cqueue.enqueue(@($cmoduleinfo.taskid, $slideid))
                }
                #
            } 
            'batch' {
                #
                $mymatch = $cqueue -match ('^' + $this.moduleinfo.project)
                if(!($mymatch -match ($this.batchid + '$'))){
                    $cqueue.enqueue(@($this.moduleinfo.project, $this.batchid))
                }
            }
            #
            default {
                #
                if (!($cqueue -match ($slideid + '$'))){
                    $cqueue.enqueue(@($this.moduleinfo.project, $slideid))
                }
                #
            }
        }
        #
    }
    <# -----------------------------------
    updatetables
    -----------------------------------
    updates the module level tables when
    modules are added \ removed or new
    projects are added \ removed. For
    each module, if the moduleobj does 
    not exist the module must be new. 
    throw an error. For new projects,
    recheck the slides for that project,
    update the module files and project status.   
    ----------------------------------- #>
    [void]updatetables($filetype){
        $this.vmq.($filetype) = $this.($filetype)
        $this.modules | & { process {
            #
            if (!$this.moduleobjs.($_)){
                $this.newmodule()
            }
            #
            $previousprojects = $this.moduleobjs.($_).module_project_data.($_) 
            $currentprojects = $this.GetAPProjects($_)
            #
            $turnedon = $currentprojects -notmatch ($previousprojects -join '|') 
            #
            if ($turnedon){
                foreach ( $newproject in $turnedon) {
                    $tasks = ($this.slide_data | & { process {
                        if ($_.project -match $newproject) {$_}
                    }}).SlideID
                    $this.addnewtasks($tasks)
                }
            }
            #
            $this.moduleobjs.($_).($filetype) = $this.($filetype)
            $this.moduleobjs.($_).updatemodulestatus($_)
            #
        }}
        #
        $this.refreshsampledb()
        #
    }
    <# -----------------------------------
    newmodule
    -----------------------------------
    if a module is added to the workflow
    all sample status should be updated. 
    Right now we have to pull down 
    ----------------------------------- #>
    [void]newmodule(){
        Throw 'A new module was detected. Support for adding modules while running does not exist'
    }
    <# -----------------------------------
    updatefullprojectdata
    -----------------------------------
    If any of the cohorts data sheets are
    updated (cohorts, paths, config) 
    import the files, check for a change in the
    module names, then update 
    the module level tables. 
    ----------------------------------- #>
    [void]updatefullprojectdata(){
        #
        $this.importcohortsinfo($this.mpath, $false)
        $this.getmodulenames($true)
        #
        $this.updatemodulestatus()
        $this.updatetables('full_project_dat')
        #
    }
    #
    [void]fileupdate($filetype){
        $storedtable = $this.getstoredtable($this.($filetype + '_data'))
        $this.('Import' + ($filetype))($this.mpath, $false)
        $tasks = $this.('changed' + $filetype)(
            $storedtable, $this.($filetype + '_data'))
        $this.addnewtasks($tasks)
        $this.updatetables($filetype + '_data')
    }
    #
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
        switch -exact ($file){
            $this.cohorts_file {
                $this.writeoutput(" cohorts file updated")
                $this.updatefullprojectdata()
            }
            $this.paths_file {
                $this.writeoutput(" paths file updated")
                $this.updatefullprojectdata()
            }
            $this.config_file {
                $this.writeoutput(" config file updated")
                $this.updatefullprojectdata()
            }
            $this.slide_file {
                $this.writeoutput(" apid file updated")
                $this.fileupdate('slide')
            }
            $this.ffmodels_file {
                $this.writeoutput(" ffmodels file updated")
                $this.fileupdate('ffmodels')
            }
            $this.corrmodels_file {
                $this.writeoutput(" corrmodels file updated")
                $this.fileupdate('corrmodels')
            }
            $this.micomp_file {
                $this.writeoutput(" micomp file updated")
                $this.fileupdate('micomp')
            }
            $this.worker_file {
                $this.writeoutput(" workers file updated")
                $this.Importworkerlist($this.mpath, $false)}
            $this.vmq.mainqueue_filename {
                $this.writeoutput(" main inform file updated")
                $this.vmq.coalescevminformqueues()
                $this.addnewtasks($this.vmq.newtasks)
                $this.vmq.newtasks = @()
                $this.refreshsampledb('vminform')
            }
            'MergeConfig' {
                $this.writeoutput(" merge config file updated")
                $this.findallantibodies()
            }
        }
        #
        foreach ($cproject in  $this.module_project_data.('vminform')){ 
            switch -exact ($fullfile){
                $this.vmq.localqueuefile.($cproject) {
                    $this.writeoutput(" local inform file updated for project: $cproject")
                    $this.vmq.coalescevminformqueues($cproject)
                    $this.addnewtasks($this.vmq.newtasks)
                    $this.vmq.newtasks = @()
                    $this.refreshsampledb('vminform')
                }
            }
        }
        #
        foreach ($cmodule in $this.modules){
            #
            switch -exact ($fullfile){
                $this.moduleobjs.($cmodule).mainqueuelocation() {
                    $this.writeoutput(" main module file updated: $cmodule")
                    $this.refreshmoduledb($cmodule)}
            }
            #
            foreach ($cproject in $this.module_project_data.($cmodule)){
                switch -exact ($fullfile){
                    $this.defprojectlogpath($cmodule, $cproject) {
                        $this.writeoutput(" logfile updated: $cmodule $cproject")
                        $this.refreshsampledb($cmodule, $cproject)}
                }
            }
            #
        }
        #
    }
    #
}