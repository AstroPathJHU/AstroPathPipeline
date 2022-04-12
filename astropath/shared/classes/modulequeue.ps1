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
    [hashtable]$localdates = @{}
    [hashtable]$localqueue = @{}
    [hashtable]$localqueuefile = @{}
    [string]$localqueue_filename
    [string]$mainqueue_filename
    [string]$mainqueue_path = '\across_project_queues'
    [string]$project
    [string]$mainqueueheaders = 'Project,Cohort,SlideID,Status,isGood'
    [string]$localqueueheaders = 'Project,Cohort,SlideID,Status,isGood'
    [string]$refobject 
    
    modulequeue($module){
        $this.modulequeueinit('\\bki04\astropath_processing', $module)
        if ($module -notcontains 'vminform'){
            $this.coalescequeues()
        }
    }
    modulequeue($mpath, $module){
        $this.modulequeueinit($mpath, $module)
        if ($module -notcontains 'vminform'){
            $this.coalescequeues()
        }
    }
    modulequeue($mpath, $module, $project){
        $this.project = $project
        $this.modulequeueinit($mpath, $module)
        if ($module -notcontains 'vminform'){
            $this.coalescequeues($project)
        }
    }
    #
    [void]modulequeueinit($mpath, $module){
        #
        $this.mpath = $mpath
        $this.module = $module
        $this.localqueue_filename = $module + '-local-queue.csv'
        $this.mainqueue_filename = $module + '-queue.csv'
        $this.refobject = 'slideid'
        #
    }
    #
  <# -----------------------------------------
     coalescequeues
     coalesce local and main  queues
     for all projects. Do not force update 
     the local queues if this option is run.
     ------------------------------------------
     Usage: $this.coalescequeues()
    ----------------------------------------- #>
    [void]coalescequeues(){
         $projects = $this.getapprojects($this.modulename)
         #
         $this.openmainqueue($false)
         $projects | ForEach-Object{
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
        if ($all){
            $this.getlocalqueue($cproject)
        } else {
            $this.getlocalqueue($cproject, $false)
        }
        $localtmp = $this.localqueue.($cproject)
        $this.localqueue.($cproject) = $this.filtertasks($cproject)
        #
        if ($localtmp){
            $cmp = Compare-Object -ReferenceObject $localtmp.($this.refobject) `
                -DifferenceObject $this.localqueue.($cproject).($this.refobject)
            if ($cmp){
                $this.writelocalqueue($cproject)
            }
         } else {
            $this.writelocalqueue($cproject)
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
        $projects = $this.getapprojects($this.modulename)
        #
        $this.openmainqueue($false)
        $projects | ForEach-Object{
            $this.coalescequeues($_, $false)
            $this.getlocalqueue($cproject, $true)
        }
        #
        $this.writemainqueue($this.mainqueuelocation())
        $this.openmainqueue($true)
        #
    }
    #
    [PSCustomObject]filtertasks($cproject){
        if ($this.module -contains 'vminform'){
            $vals = $this.maincsv | where-object {$_.taskid -match ('T' + $cproject.PadLeft(3,'0'))}
        } else {
            $vals = ($this.maincsv.project -contains $cproject)
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
            if ($this.module -match 'vminform'){
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
    }
    <# -----------------------------------------
     openmainqueue
     open main inform queue
     ------------------------------------------
     Usage: $this.openmainqueue()
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
            $this.setfile($this.localqueuefile.($project), `
               $this.localqueueheaders)
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
        }
        #
        $this.localqueue.($project) = $q
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
            '\' + $cohortinfo.Dname + '\upkeep_and_progress'
            $localqueuepath = $this.CrossPlatformPaths($localqueuepath)
        return $localqueuepath
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
    #
    [void]writelocalqueue($project){
        #
        $updatedlocal = (($this.localqueue.($project) | 
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
    [void]writemainqueue($mainqueuelocation){
        #
        [array]$heads = ($this.mainqueueheaders -split ',')
        $mainqueue = $this.maincsv | 
            select-object -Property $heads
        #
        $updatedmain = (($mainqueue |
            ConvertTo-Csv -NoTypeInformation) -join "`r`n").Replace('"','') + "`r`n"
        #
        try {
            $this.UnregisterEvent($this.mainqueuelocation())
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
            $this.FileWatcher($this.mainqueuelocation())
        }
        #
    }
    #
}
