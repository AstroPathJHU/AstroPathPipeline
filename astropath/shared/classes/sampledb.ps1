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
    [switch]$wfon = $true
    #
    [string]$vminform_nonab_keys = (@('project','slidelog','mainlog','starttime',
        'finishtime','version','status') -join '|')
    #
    [hashtable]$status_settings
    #
    sampledb(): base ('\\bki04\astropath_processing'){
        $this.sampledbinit()
    }
    sampledb($mpath) : base ($mpath) {ZW
        $this.sampledbinit()
    }
    sampledb($mpath, $projects) : base ($mpath){
        $this.projects = $projects
        $this.sampledbinit()
    }
    #
    sampledbinit(){
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
            $c = $this.updatesample($c, $ctotal, $slide.slideid)
        }
        #
        $this.progressbarfinish()
        $this.writeoutput(" Sample status database built")
        #
    }
    #
    [void]defsampleStages($project){
        #
        $c = 1
        $ctotal = ($this.slide_data |
            Where-Object {$_.project -contains $project}).count
        $this.writeoutput(" Building sample status database")
        #
        $this.slide_data |
            Where-Object {$_.project -contains $project} | 
            & { process {
                $c = $this.updatesample($c, $ctotal, $_.slideid)
            }}
        #
        $this.progressbarfinish()
        $this.writeoutput(" Sample status database built")
        #
    }
    #
    [int]updatesample($c, $ctotal, $slideid){
        #
        $this.progressbar($c, $ctotal, ($slideid, 'update sample status' -join ' - '))
        $this.preparesample($slideid, $c, $ctotal)
        $this.progressbar($c, $ctotal, ($slideid, 'update module table' -join ' - '))
        $this.defmoduletables()
        $c += 1 
        return $c
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
    [void]defmoduletables(){
        #
        $this.modules | & { process {
            $this.refreshmoduledb($_, $true)
        }}
        #
    }
    <# -----------------------------------------
    # refresh all the module tables for a particular sample
    # runs if a logging 'newtask' is created (i.e. a finish event)
    ----------------------------------------- #>
    [void]refreshsampledb(){
        #
        $this.writeoutput("     refreshing sample data base for new tasks started")
        #
        if ($this.newtasks){
            $new = $true
        } else {
            $new = $false
        }
        #
        $ctotal = $this.newtasks.count
        $c = 1
        #
        while($this.newtasks){
            #
            $slide, $this.newtasks = $this.newtasks
            $c = $this.updatesample($c, $ctotal, $slide)
            #
        }
        #
        if ($new){
            $this.progressbarfinish()
        }
        #
        $this.writeoutput("     refreshing sample data base for new tasks finished")
        #
    }
    #
    [void]refreshsampledb($cmodule){
        #
        $ctotal = $this.newtasks.count
        $c = 1
        #
        if ($this.newtasks){
            $this.writeoutput("     refreshing sample data base for [$cmodule] started")
            $new = $true
        } else {
            $new = $false
        }
        #
        while($this.newtasks){
            $slide, $this.newtasks = $this.newtasks
            $c = $this.updatesample($c, $ctotal, $slide)
            $this.moduleobjs.($cmodule).writelocalqueue(
                $this.project)
        }
        #
        if ($new){
            $this.progressbarfinish()
            $this.writeoutput("     refreshing sample data base for [$cmodule] finished")
        }
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
        $this.writeoutput("     updating module database: [$cmodule] started")
        $ctotal = $this.moduleobjs.($cmodule).newtasks.count
        $c = 1
        #
        $this.moduleobjs.($cmodule).openmainqueue($false)
        if ($this.moduleobjs.($cmodule).newtasks){
            $new = $true
        } else {
            $new = $false
        }
        #
        while($this.moduleobjs.($cmodule).newtasks){
            $slide, $this.moduleobjs.($cmodule).newtasks =
                $this.moduleobjs.($cmodule).newtasks
            $c = $this.updatesample($c, $ctotal, $slide)
        }
        #
        if ($new){
            $this.progressbarfinish()
        }
        #
        $this.moduleobjs.($cmodule).writemoduledb()
        $this.moduleobjs.($cmodule).coalescequeues()
        #
        if ($this.moduleobjs.($cmodule).newtasks){
            $this.refreshmoduledb($cmodule)
        }
        $this.writeoutput("     updating module database: [$cmodule] finished")
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
    [void]refreshmoduledb($cmodule, $isslide){
        #
        if ($cmodule -match 'vminform'){
            $this.refreshmoduledbsubvm($cmodule)
        } else {
            $this.refreshmoduledbsub($cmodule)
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
    [void]refreshmoduledbsub($cmodule){
        #
        $row = $this.moduleobjs.($cmodule).maincsv |
         & { process {
            if ($_.slideid -contains $this.slideid) { $_ }
        }}
        if ($row){
            $this.updatemoduleline($cmodule)
        } else {
            $this.createmoduleline($cmodule)
        }
        #
    }
    #
    [void]refreshmoduledbsubvm($cmodule){
        #
        $row = $this.moduleobjs.($cmodule).localqueue.($this.project) |
         & { process {
            if ($_.slideid -contains $this.slideid) { $_ }
        }}
        if ($row){
            $this.updatemodulelinevm($cmodule)
        } else{
            $this.createmodulelinevm($cmodule)
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
    [void]createmoduleline($cmodule){
        #
        $currentobj = $this.moduleinfo.($cmodule)
        #
        $row = $this.buildbaserow()
        $row | Add-Member -NotePropertyMembers @{
                Status = ($currentobj.status -replace ',', ';')
                isGood = 1
                StartTime = $currentobj.StartTime
                FinishTime = $currentobj.FinishTime
            } -PassThru
        #
        if(!$this.moduleobjs.($cmodule).maincsv){
            $this.moduleobjs.($cmodule).maincsv = @()
        }
        $this.moduleobjs.($cmodule).maincsv += $row 
        #
        if ($currentobj.status -match $this.status.ready){
            $this.enqueuetask($cmodule, $currentobj)
        }
        #
    }
    #
    [void]createmodulelinevm($cmodule){
        #
        if(!$this.moduleobjs.($cmodule).localqueue.($this.project)){
            $this.moduleobjs.($cmodule).localqueue.($this.project) = @()
        }
        #
        $this.getantibodies($this.project)
        #
        $row = $this.buildbaserow()
        #
        $this.antibodies | & { process {
            #
            $currentobj = $this.moduleinfo.($cmodule).($_)
            $status = $currentobj.status -replace ',', ';'
            #
            if ($status -match $this.status.ready){
                $this.enqueuetask($cmodule, $currentobj)
            }
            #
            $row | Add-Member -NotePropertyMembers @{
                ($_ + '_Status') =  $status
                ($_ + '_StartTime') =  $currentobj.StartTime
                ($_ + '_FinishTime') =  $currentobj.FinishTime
                ($_ + '_Algorithm') = $currentobj.algorithm
            } -PassThru
        }}
        #
        $this.moduleobjs.($cmodule).localqueue.($this.project) += $row
        #
    }
    #
    [PSCustomObject]buildbaserow(){
        #
        $row = [PSCustomObject]@{
            Project = $this.project
            Cohort = $this.cohort
            BatchID = $this.batchid
            SlideID = $this.slideid
        }
        #
        return $row
    }
    <# -----------------------------------------
    updatemoduleline    
    -----------------------------------------
    updates the preexisting module lines for the
    specified slide. overload (3) to include the
    project for vminform task
    ----------------------------------------- #>
    [void]updatemoduleline($cmodule){
        #
        $row = $this.moduleobjs.($cmodule).maincsv |
         & { process {
            if ($_.slideid -contains $this.slideid) { $_ }
        }}
        #
        $statlabel = 'Status'
        $startlabel = 'StartTime'
        $finishlabel = 'FinishTime'
        #
        $cmoduleinfo = $this.moduleinfo.($cmodule)
        #
        $this.updatemodulesub($cmodule, $row, $cmoduleinfo,
            $statlabel, $startlabel, $finishlabel)
        #
    }
    #
    [void]updatemodulelinevm($cmodule){
        #
        $this.getantibodies($this.project)
        $row = $this.moduleobjs.($cmodule).localqueue.($this.project) |
         & { process {
            if ($_.slideid -contains $this.slideid) { $_ }
        }}
        #
        $this.antibodies | & { process {
            #
            $statlabel = ($_ + '_Status')
            $algname = ($_ + '_Algorithm')
            $startlabel = ($_ + '_StartTime')
            $finishlabel = ($_ + '_FinishTime')
            #
            $cmoduleinfo = $this.moduleinfo.($cmodule).($_)
            $row.($algname) = $cmoduleinfo.algorithm
            #
            $this.updatemodulesub($cmodule, $row, $cmoduleinfo,
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
    [void]updatemodulesub($cmodule, $row, $cmoduleinfo, $statlabel,
        $startlabel, $finishlabel){
        #
        $slidestatus = $cmoduleinfo.status -replace ',', ';'
        $modulestatus = $row.($statlabel)
        #
        if ($slidestatus -match $this.status.ready){
            $this.enqueuetask($cmodule, $cmoduleinfo)
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
            $this.enqueuetask($cmodule, $cmoduleinfo)
            $row.($statlabel) = $this.status.ready
            $cmoduleinfo.StartTime = $this.empty_time
            $cmoduleinfo.FinishTime = $this.empty_time
        }
        #
        if (
            ($row.($startlabel) -ne $cmoduleinfo.StartTime)
        ) {
            $row.($startlabel) = $cmoduleinfo.StartTime
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
    [void]enqueuetask($cmodule, $cmoduleinfo){
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
        switch -regex ($cmodule){
            'vminform' {
                #
                if ($cmoduleinfo.taskid -and 
                    !($cqueue -match ('^' + $cmoduleinfo.taskid))){
                    $cqueue.enqueue(@($cmoduleinfo.taskid, $this.slideid))
                }
                #
            } 
            'batch' {
                #
                if ($this.checkbatch($cmodule)){
                    $mymatch = $cqueue -match ('^' + $this.moduleinfo.project)
                    if(!($mymatch -match ($this.batchid + '$'))){
                        $cqueue.enqueue(@($this.moduleinfo.project, $this.batchid))
                    }
                }
            }
            #
            default {
                #
                if (!($cqueue -match ($this.slideid + '$'))){
                    $cqueue.enqueue(@($this.moduleinfo.project, $this.slideid))
                }
                #
            }
        }
        #
    }
    #
    [switch]checkbatch($cmodule){
        #
        if (
            ($this.moduleobjs.($cmodule).maincsv |
                & { process {
                    if (
                        $_.project -contains $this.project.trim() -and 
                        $_.batchid -contains $this.batchid.trim()
                    ) {$_}}}
            ).status -notmatch $this.status.ready
        ) { return $false }
        #
        return $true
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
        Throw (
            'A new module was detected.',
            'Support for adding modules',
            'while running does not exist'
        ) -join ' '
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
        $fullfile = ($fullfile -split ';')[0]
        #
        start-sleep 2
        #
        $fpath = Split-Path $fullfile
        $file = Split-Path $fullfile -Leaf
        #
        switch -exact ($file){
            $this.cohorts_file {
                $this.writeoutput(" cohorts file updated")
                $this.updatefullprojectdata()
                $this.writeoutput(" cohorts file checks finished")
            }
            $this.paths_file {
                $this.writeoutput(" paths file updated")
                $this.updatefullprojectdata()
                $this.writeoutput(" paths file checks finished")
            }
            $this.config_file {
                $this.writeoutput(" config file updated")
                $this.updatefullprojectdata()
                $this.writeoutput(" config file checks finished")
            }
            $this.slide_file {
                $this.writeoutput(" apid file updated")
                $this.fileupdate('slide')
                $this.writeoutput(" apid file checks finished")
            }
            $this.ffmodels_file {
                $this.writeoutput(" ffmodels file updated")
                $this.fileupdate('ffmodels')
                $this.writeoutput(" ffmodels file checks finished")
            }
            $this.corrmodels_file {
                $this.writeoutput(" corrmodels file updated")
                $this.fileupdate('corrmodels')
                $this.writeoutput(" corrmodels file checks finished")
            }
            $this.micomp_file {
                $this.writeoutput(" micomp file updated")
                $this.fileupdate('micomp')
                $this.writeoutput(" micomp file checks finished")
            }
            $this.worker_file {
                $this.writeoutput(" workers file updated")
                $this.Importworkerlist($this.mpath, $false)
                $this.writeoutput(" workers file checks finished")
            }
            $this.vmq.mainqueue_filename {
                $this.writeoutput(" main inform file updated")
                $this.vmq.coalescevminformqueues()
                $this.addnewtasks($this.vmq.newtasks)
                $this.vmq.newtasks = @()
                $this.refreshsampledb('vminform')
                $this.writeoutput(" main inform file checks finished")
            }
            'MergeConfig' {
                $this.writeoutput(" merge config file updated")
                $this.findallantibodies()
                $this.writeoutput(" merge config file checks finished")
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
                    $this.writeoutput(" local inform file checks finished for project: $cproject")
                }
            }
        }
        #
        foreach ($cmodule in $this.modules){
            #
            switch -exact ($fullfile){
                $this.moduleobjs.($cmodule).mainqueuelocation() {
                    $this.writeoutput(" main module file updated for [$cmodule]")
                    $this.refreshmoduledb($cmodule)
                    $this.writeoutput(" main module file check finished for [$cmodule]")
                }
            }
            #
            foreach ($cproject in $this.allprojects){
                switch -exact ($fullfile){
                    $this.defprojectlogpath($cmodule, $cproject) {
                        $this.writeoutput(" logfile updated for [$cmodule] - project: $cproject")
                        $this.writeoutput(" checking log for new tasks [$cmodule] - project: $cproject")
                        $this.refreshsampledb($cmodule, $cproject)
                        $this.writeoutput(" logfile checks finished for [$cmodule] - project: $cproject")
                    }
                }
            }
            #
        }
        #
    }
    #
}