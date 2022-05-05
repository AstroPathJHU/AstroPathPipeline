<# -------------------------------------------
 modulequeue
 created by: Benjamin Green, Andrew Joqurea - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 a special case of the queue functionality
 -------------------------------------------#>
 class modulequeue : sharedtools {
    #
    [array]$maincsv
    [array]$lastopenmaincsv
    [hashtable]$localdates = @{}
    [hashtable]$localqueue = @{}
    [hashtable]$lastopenlocalqueue = @{}
    [hashtable]$localqueuefile = @{}
    [string]$localqueue_filename
    [string]$mainqueue_filename
    [string]$mainqueue_path = '\across_project_queues'
    [string]$localqueue_path = '\progress_tables'
    [string]$project
    [string]$mainqueueheaders = 'Project,Cohort,SlideID,Status,isGood,StartTime,FinishTime'
    [string]$localqueueheaders = 'Project,Cohort,SlideID,Status,isGood,StartTime,FinishTime'
    [string]$refobject 
    [string]$type = 'table'
    #
    modulequeue(){
        $this.modulequeueinit('\\bki04\astropath_processing', '')
        #
    }
    #
    modulequeue($module){
        $this.modulequeueinit('\\bki04\astropath_processing', $module)
        #
        if ($this.type -notcontains 'queue'){
            $this.coalescequeues()
        }
        #
    }
    modulequeue($mpath, $module){
        $this.modulequeueinit($mpath, $module)
        #
        if ($this.type -notcontains 'queue'){
            $this.coalescequeues()
        }
        #
    }
    modulequeue($mpath, $module, $project){
        $this.project = $project
        $this.modulequeueinit($mpath, $module)
        #
        if ($this.type -notcontains 'queue'){
            $this.coalescequeues($project)
        }
        #
    }
    #
    [void]modulequeueinit($mpath, $module){
        #
        $this.mpath = $mpath
        $this.module = $module
        $this.localqueue_filename = $module + '-local-'+ $this.type+'.csv'
        $this.mainqueue_filename = $module + '-'+ $this.type+'.csv'
        $this.refobject = 'slideid'
        #
    }
  <# -----------------------------------------
     coalescequeues
     coalesce local and main  queues
     for all projects. Do not force update 
     the local queues if this option is run.
     ------------------------------------------
     Usage: $this.coalescequeues()
    ----------------------------------------- #>
    [void]coalescequeues(){
         #
         $this.getmodulestatus($this.module)
         #
         $this.openmainqueue($false)
         #
         $this.allprojects | ForEach-Object{
            $this.coalescequeues($_, $true)
         }
         #
         $this.writemainqueue($this.mainqueuelocation())
         #
    }
  <# -----------------------------------------
     coalescequeues
     coalesce local and main queues
     for a particular project. Force updates
     the local queue but do not create a new
     file watcher. 
     ------------------------------------------
     Usage: $this.coalescequeues($project)
    ----------------------------------------- #>
    [void]coalescequeues($project){
        #
        $this.openmainqueue($false)
        $this.coalescequeues($project, $false)
        $this.writemainqueue($this.mainqueuelocation())
        #
    }
    <# -----------------------------------------
    # if all is false it will force the local queue to update
    # this is not needed on startup or when the main queue is
    # the one that has been updated (when coalesce is run w/o args)
    ----------------------------------------- #>
    [void]coalescequeues($project, $all){
        #
        $cproject = $project.ToString()
        if ($this.coalesceinformtables($cproject)){
            return
        }
        #
        if ($all){
            $this.getlocalqueue($cproject)
        } else {
            $this.getlocalqueue($cproject, $false)
        }
        #
        $localtmp = $this.getstoredtable($this.localqueue.($cproject))
        $this.localqueue.($cproject) = $this.filtertasks($cproject)
        $this.comparewrite($cproject, $localtmp)
        #
    }
    #
    [switch]coalesceinformtables($project){
        #
        if ($this.module -match 'vminform' -and 
            $this.type -match 'table') {
            #
            $localtmp = $this.getstoredtable($this.localqueue.($project))
            $this.getlocalqueue($project, $false)
            $localtmp2 = $this.getstoredtable($this.localqueue.($project))
            $this.localqueue.($project) = $localtmp
            #
            $this.comparewrite($project, $localtmp2)
            return $true
        }
        #
        return $false
        #
    }
    #
    [void]comparewrite($project, $localtmp){
        #
        if ($this.localqueue.($project)){
            if ($localtmp){
                $cmp = Compare-Object -ReferenceObject $localtmp.($this.refobject) `
                    -DifferenceObject $this.localqueue.($project).($this.refobject)
                if ($cmp){
                    $this.writelocalqueue($project)
                }
            } else {
                $this.writelocalqueue($project)
            }
        }
        #
    }
    #
    [PSCustomObject]filtertasks($cproject){
        if ($this.type -contains 'queue'){
            $vals = $this.maincsv |
                where-object {$_.taskid -match ('T' + $cproject.PadLeft(3,'0'))}
        } else {
            $vals = $this.maincsv | 
                Where-Object {$_.project -contains $cproject}
        }
        return $vals
    }
    <# -----------------------------------------
     mainqueuelocation
     open main inform queue
     ------------------------------------------
     Usage: $this.mainqueuelocation()
    ----------------------------------------- #>
    [string]mainqueuelocation(){
        #
        $mainqueuefile = $this.mpath +
             $this.mainqueue_path + '\' + $this.mainqueue_filename
        #
        return $mainqueuefile
        #
    } 
    <# -----------------------------------------
     openmainqueue
     open main inform queue
     ------------------------------------------
     Usage: $this.openmainqueue()
    ----------------------------------------- #>
    [void]openmainqueue(){
        #
        if (!$this.maincsv){
            $this.openmainqueue($false)
        }
        #
    }
    #
    [void]openmainqueue($createwatcher){
        #
        if ($this.module -match 'vminform' -and 
            $this.type -match 'table') {
            return
        }
        #
        $mainqueuefile = $this.mainqueuelocation()
        #
        if (!(test-path $mainqueuefile)){
            $this.setfile($mainqueuefile,
                $this.mainqueueheaders)
        }
        #
        [array]$mainqueue = $this.OpenCSVFile($mainqueuefile)
        if ($mainqueue){
            $mainqueue = $mainqueue | where-object { 
                $_.($this.refobject) -and $_.($this.refobject).trim().length -gt 0
            }
        }
        #
        if ($mainqueue){
            if ($this.type -match 'queue'){
                $mainqueue | foreach-object {
                    $_ | Add-Member localtaskid $_.taskid.substring(4).trimstart('0') -PassThru
                }
            }
        }
        #
        if ($createwatcher){
            $this.FileWatcher($mainqueuefile)
        }
        #
        $this.maincsv = $mainqueue
        #
        if($this.lastopenmaincsv){
            $this.getnewtasksmain($this.lastopenmaincsv, $this.maincsv)
        }
        #
        $this.lastopenmaincsv = $this.getstoredtable($mainqueue)
        #
    }
    #
    [void]getlocalqueue($project){
        #
        $this.localqueuefile.($project) = $this.localqueuelocation($project)
        $this.openlocalqueue($project)
        #
    }
    #
    [void]getlocalqueue($project, $createwatcher){
        #
        $this.localqueuefile.($project) = $this.localqueuelocation($project)
        $this.openlocalqueue($project, $createwatcher)
        #
    }
    <# -----------------------------------------
     localqueuelocation
     open local inform queue
     ------------------------------------------
     Usage: $this.localqueuelocation()
    ----------------------------------------- #>
    [string]localqueuelocation($project){
        #
        $localqueuefilea = $this.localqueuepath($project) +
            '\' + $this.localqueue_filename
        #
        return $localqueuefilea
        #
    }
    #
    [string]localqueuepath($project){
        $cohortinfo = $this.GetProjectCohortInfo($this.mpath, $project)
        $localqueuepath = $cohortinfo.Dpath +
            '\' + $cohortinfo.Dname + '\upkeep_and_progress' + $this.localqueue_path
        #
        $localqueuepath = $this.CrossPlatformPaths(
                $this.uncpaths($localqueuepath)
        )
        return $localqueuepath
    }
    <# -----------------------------------------
     openlocalqueue
     open local module queue table
     ------------------------------------------
     Usage: $this.openlocalqueue()
    ----------------------------------------- #>
    [void]openlocalqueue($project){
        #
        if (!($this.localqueue.($project))){
            $this.openlocalqueue($project, $false)
        }
        #
    }
    #
    [void]openlocalqueue($project, $createwatcher){
        #
        if (!(test-path $this.localqueuefile.($project))){
            #
            if ($this.module -match 'vminform' -and 
                $this.type -match 'table') {
                $headers = $this.createvmtableheaders($project)
                $headers = ($headers -join ',')
            } else {
                $headers = $this.localqueueheaders
            }
            #
            $this.setfile($this.localqueuefile.($project), `
                $headers)
        }
        #
        if ($createwatcher){
            $this.FileWatcher($this.localqueuefile.($project))
        }
        #
        [array]$q = $this.OpenCSVFile($this.localqueuefile.($project))
        #
        if ($q){
            $q = $q | where-object { 
                $_.($this.refobject) -and $_.($this.refobject).trim().length -gt 0
            }
            if ($this.type -match 'queue'){
                $q | foreach-object {
                    $_ | Add-Member localtaskid $_.taskid -PassThru
                }
            }
        }
        #
        $this.localqueue.($project) = $q
        $this.lastopenlocalqueue.($project) = $this.getstoredtable($q)
        #
    }  
    #
    [void]writelocalqueue($project){
        #
        if (!($this.localqueue.($project) )){
            return
        }
        #
        if ($this.module -match 'vminform' -and 
            $this.type -match 'table') {
            $heads = $this.createvmtableheaders($project)
        } else {
            $heads = $this.localqueueheaders -split ','
        }
        #
        $updatedlocal = (($this.localqueue.($project) | 
            select-object -Property $heads | 
            ConvertTo-Csv -NoTypeInformation) -join "`r`n").Replace('"','') + "`r`n"
        try {
            $this.UnregisterEvent($this.localqueuefile.($project))
            $isevent = $true
        } catch {
            $isevent = $false
        }
        #
        try{
            $this.SetFile($this.localqueuefile.($project), $updatedlocal)
        } catch {}
        #
        if ($isevent){
            $this.FileWatcher($this.localqueuefile.($project))
        }
        #
    }
    #
    [void]writemainqueue(){
        $this.writemainqueue($this.mainqueuelocation())
    }
    #
    [void]writemainqueue($mainqueuelocation){
        #
        if ($this.module -match 'vminform' -and 
            $this.type -match 'table') {
            return
        }
        #
        if (!($this.maincsv )){
            return
        }
        #
        [array]$heads = ($this.mainqueueheaders -split ',')
        #
        $stable = $this.getstoredtable($this.lastopenmaincsv)
        $this.lastopenmaincsv = ''
        #
        $mainqueue = $this.getstoredtable($this.maincsv)
        $this.openmainqueue($false)
        $this.getnewtasksmain($stable, $this.lastopenmaincsv)
        #
        if ($this.module -match 'vminform'){
            if ($this.comparetablesvmmain($mainqueue, $this.maincsv)){
                $this.maincsv = $mainqueue
                $this.lastopenmaincsv = $this.getstoredtable($this.maincsv) 
            } else {
                return
            }
        } else {
            if ($this.comparetables($mainqueue, $this.maincsv)){
                $this.maincsv = $mainqueue
                $this.lastopenmaincsv = $this.getstoredtable($this.maincsv) 
            } else {
                return
            }
        }
        #
        $mainqueue = $this.maincsv | 
            select-object -Property $heads
        #
        $updatedmain = (($mainqueue |
            ConvertTo-Csv -NoTypeInformation) -join "`r`n").Replace('"','') + "`r`n"
        #
        try {
            $this.UnregisterEvent($mainqueuelocation)
            $isevent = $true
        } catch {
            $isevent = $false
        }
        #
        try {
            $this.SetFile($mainqueuelocation, $updatedmain)
        } catch {}
        #
        if ($isevent){
            $this.FileWatcher($mainqueuelocation)
        }
        #
    }
    #
    <#
        If a module queue is updated locally, we need to get the 
        slideids from that queue so that we can update their status
        and enque the corresponding task to the module worker queues
    #>
    [void]getnewtasksmain($oldqueue, $newqueue){
        #
        if (!$this.newtasks){
            $this.newtasks = @()
        }
        #
        if (!$newqueue){
            return
        }
        #
        if (!$oldqueue){
            return
        }
        #
        if ($this.type -match 'queue'){
            #
            $cmp = Compare-Object -ReferenceObject $newqueue `
                -DifferenceObject $oldqueue -Property 'TaskID', 'SlideID', 'Antibody', 'Algorithm'
            #
        } else {
            #
            $cmp = Compare-Object -ReferenceObject $newqueue `
                -DifferenceObject $oldqueue -Property 'SlideID', 'Status', 'StartTime', 'FinishTime'
            #
        }
        #
        if ($cmp){
            #
            $this.newtasks += ($cmp |
                Where-Object {$_.SideIndicator -match '<='}).SlideID
            #
        }
        #
    }
    #
    [switch]comparetables($oldqueue, $newqueue){
        #
        if (!$oldqueue){
            return $false
        }
        #
        if (!$newqueue){
            return $true
        }
        #
        if ($this.type -match 'queue'){
            #
            $cmp = Compare-Object -ReferenceObject $newqueue `
                -DifferenceObject $oldqueue -Property 'TaskID', 'SlideID'
            #
        } else {
            #
            $cmp = Compare-Object -ReferenceObject $newqueue `
                -DifferenceObject $oldqueue -Property 'SlideID', 'Status', 'StartTime', 'FinishTime'
            #
        }
        #
        if ($cmp){
            return $true
        } else {
            return $false
        }
        #
    }
    #
    [switch]comparetablesvmmain($oldqueue, $newqueue){
        #
        if (!$oldqueue){
            return $false
        }
        #
        if (!$newqueue){
            return $true
        }
        #
        if ($this.type -match 'queue'){
            #
            $cmp = Compare-Object -ReferenceObject $newqueue `
                -DifferenceObject $oldqueue -Property 'TaskID', 'SlideID', 'ProcessingLocation', 'StartDate'
            #
        } else {
            #
            $cmp = Compare-Object -ReferenceObject $newqueue `
                -DifferenceObject $oldqueue -Property 'SlideID', 'Status', 'StartTime', 'FinishTime'
            #
        }
        #
        if ($cmp){
            return $true
        } else {
            return $false
        }
        #
    }
    <#------------------------------------------
    will force update all local and main 
    queues and create file watchers. to run after the
    sampledb has been built
    --------------------------------------------#>
    [void]createwatchersqueues(){
        #
        if ($this.type -contains 'queue'){
            return
        }
        #
        $this.getmodulestatus($this.module)
        #
        $this.writemainqueue()
        $this.openmainqueue($true)
        $this.allprojects | ForEach-Object{
            $this.coalescequeues($_, $false)
            $this.getlocalqueue($_, $true)
        }
        #
    }
    #
    [array]createvmtableheaders($project){
        #
        $this.importcohortsinfo($this.mpath) | Out-Null
        #
        $project_dat = $this.full_project_dat| 
                    Where-Object -FilterScript {$_.Project -eq $project}
        #
        $root = $this.uncpaths($project_dat.dpath)
        #
        $basepath = $root, $project_dat.dname -join '\'
        #
        $this.findantibodies($basepath)
        #
        $headers = @('Project','Cohort','SlideID')
        #
        $this.antibodies | ForEach-Object {
            $statusname = ($_ + '_Status')
            $startname = ($_ + '_StartTime') 
            $finishname = ($_ + '_FinishTime')
            $headers += @($statusname, $startname, $finishname)
        }
        #
        return $headers
        #
    }
    #
}
