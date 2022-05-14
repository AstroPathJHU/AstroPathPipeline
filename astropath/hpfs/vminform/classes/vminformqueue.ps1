<# -------------------------------------------
 vminformqueue
 created by: Benjamin Green, Andrew Joqurea - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 a special case of the module queue 
 -------------------------------------------#>
class vminformqueue : modulequeue {
    #
    [string]$informvers = '2.4.8'
    [string]$type = 'queue'
    [PSCustomObject]$localslide
    [PSCustomObject]$mainslide
    #
    vminformqueue() : base ('vminform'){
        $this.vminformqueueinit()
        $this.coalescevminformqueues()
        $this.newtasks = @()
    }
    vminformqueue($mpath): base ($mpath, 'vminform'){
        $this.vminformqueueinit()
        $this.coalescevminformqueues()
        $this.newtasks = @()
    }
    vminformqueue($mpath, $project) : base($mpath, 'vminform', $project){
        $this.vminformqueueinit()
        $this.coalescevminformqueues($project)
        $this.newtasks = @()
    }
    vminformqueueinit(){
        $this.refobject = 'taskid'
        $this.localqueue_filename = 'inForm_queue.csv'
        $this.mainqueueheaders = 'TaskID,slideid,Antibody,Algorithm,ProcessingLocation,StartDate'
        $this.localqueueheaders = 'TaskID,slideid,Antibody,Algorithm'
    }
  <# -----------------------------------------
     coalescevminformqueues
     coalesce local and main vminform queues
     for all projects. Do not force update 
     the local queues if this option is run.
     ------------------------------------------
     Usage: $this.coalescevminformqueues()
    ----------------------------------------- #>
    [void]coalescevminformqueues(){
        #
        $this.getmodulestatus($this.module)
        #
        $this.openmainqueue($false)
        $this.allprojects | & { process {
            $this.coalescevminformqueues($_, $true)
        }}
        #
        if($this.lastopenmaincsv){
            $this.getnewtasksmain($this.lastopenmaincsv, $this.maincsv)
        }
        #
        $this.writemainqueue($this.mainqueuelocation())
        #
    }
  <# -----------------------------------------
     coalescevminformqueues
     coalesce local and main vminform queues
     for a particular project. Force updates
     the local queue but do not create a new
     file watcher. 
     ------------------------------------------
     Usage: $this.coalescevminformqueues($project)
    ----------------------------------------- #>
    [void]coalescevminformqueues($project){
        #
        $this.openmainqueue()
        $this.coalescevminformqueues($project, $false)
        if($this.lastopenmaincsv){
            $this.getnewtasksmain($this.lastopenmaincsv, $this.maincsv)
        }
        $this.writemainqueue($this.mainqueuelocation())
        #
    }
    <# -----------------------------------------
    # if all is false it will force the local queue to update
    # this is not needed on startup or when the main queue is
    # the one that has been updated (when coalesce is run w/o args)
    ----------------------------------------- #>
    [void]coalescevminformqueues($project, $all){
        #
        $this.openmainqueue()
        #
        $cproject = $project.ToString()
        if ($all){
            $this.getlocalqueue($cproject)
        } else {
            $this.getlocalqueue($cproject, $false)
        }
        $localtmp =  $this.getstoredtable($this.localqueue.($cproject))
        $this.pairqueues($cproject)
        #
        if ($localtmp){
            if ( 
                Compare-Object -ReferenceObject $localtmp.taskid `
                -DifferenceObject $this.localqueue.($cproject).taskid
            ){
                $this.writelocalqueue($cproject)
            }
         } else {
            $this.writelocalqueue($cproject)
        }
        #
        $localtmp = $null
        #
    }
    <#------------------------------------------
    will force update all local and main inform 
    queues and create file watchers. to run after the
    sampledb has been built
    --------------------------------------------#>
    [void]createwatchersvminformqueues(){
        #
        $this.getmodulestatus($this.module)
        #
        $this.writemainqueue()
        $this.openmainqueue($false)
        $this.allprojects | & { process {
            #
            $this.coalescevminformqueues($_, $false)
            $this.getlocalqueue($_, $true)
            #
        }}
        #
        if($this.lastopenmaincsv){
            $this.getnewtasksmain($this.lastopenmaincsv, $this.maincsv)
        }
        #
        $this.writemainqueue()
        $this.openmainqueue($true)
        #
    }
    #
    [void]pairqueues($cproject){
        #
        $currentprojecttasks = $this.filtertasks($cproject)
        $this.updatelocalvminformqueue($currentprojecttasks, $cproject)
        $this.updatemainvminformqueue($currentprojecttasks, $cproject)
        $currentprojecttasks = $null
        $cproject = $null
        #
    }
    <# -----------------------------------------
     updatelocalvminformqueue
     update local vminform queue
     ------------------------------------------
     Usage: $this.updatelocalvminformqueue()
    ----------------------------------------- #>
    [void]updatelocalvminformqueue($currentprojecttasks, $project){
        #
        if (!$this.localqueue.($project)){
            $this.localqueue.($project) = @()
        }
        #
        if (!$currentprojecttasks){
            return
        }
        #
        $nexttaskid = ($currentprojecttasks.localtaskid |
            measure-object -maximum).Maximum + 1
        #
        $differenttasks = (Compare-Object -ReferenceObject $currentprojecttasks `
            -DifferenceObject $this.localqueue.($project) `
            -Property 'localtaskid','SlideID','Antibody','Algorithm' ) | 
        & { process {
            if ($_.SideIndicator -match '<='){ $_ }}}
        #
        $differenttasks | & { process {
            #
            $this.addtolocal($project, $_)
            $nexttaskid = $this.correctduplicates($project, $_.localtaskid, $nexttaskid)
            $nexttaskid = $this.correctmismatches($project, $_, $nexttaskid)
            #
        }}
        #
    }
    #
    # tasks in main not in local
    #
    [void]addtolocal($project, $row){
        if ($row.localtaskid -notin $this.localqueue.($project).TaskID) {
            $this.localqueue.($project) += $row |
                    select-object -Property slideid, Antibody, Algorithm, localtaskid |
                    Add-Member TaskID $row.localtaskid -PassThru
        }
    }
    #
    # if taskid in main matches more than 1 in the local, replace
    # local with new taskids until it only matches 1
    #
    [int]correctduplicates($project, $localtaskid, $nexttaskid){
        #
        $localrow = $this.selectlocalrows($project, $localtaskid)
        #
        while ($localrow.count -gt 1){
            $localrow[1].TaskID = [string]$nexttaskid
            $localrow[1].localtaskid = [string]$nexttaskid
            $nexttaskid ++
            $localrow = $this.selectlocalrows($project, $localtaskid)
        }
        #
        return $nexttaskid
        #
    }
    #
    [int]correctmismatches($project, $row, $nexttaskid){
        #
        $localrow = $this.selectlocalrows($project, $row.localtaskid)
        #
        if (($localrow |
                    select-object -Property slideid, Antibody, Algorithm) `
                -notmatch ($row |
                    select-object -Property slideid, Antibody, Algorithm)
            ) {
                #
                $NewRow =  $localrow |
                    select-object -Property slideid, Antibody, Algorithm |
                    Add-Member  -NotePropertyMembers @{
                        TaskID = [string]$nexttaskid
                        localtaskid = [string]$nexttaskid
                     } -PassThru
                $this.localqueue.($project) += $NewRow
                #
                $localrow[0].slideid = $row.slideid
                $localrow[0].Antibody = $row.Antibody
                $localrow[0].Algorithm = $row.Algorithm
                $localrow[0].localtaskid = $row.localtaskid
                $nexttaskid ++
            }
        #
        return $nexttaskid
        #
    }
    #
    [PSCustomObject]selectlocalrows($project, $localtaskid){
        #
        return (
            $this.localqueue.($project) |
                & { process {
                    if ($_.TaskID -eq $localtaskid) { $_ }
                }}
        )
        #
    }
    #
    <# -----------------------------------------
     updatemainvminformqueue
     update main vminform queue
     ------------------------------------------
     Usage: $this.updatemainvminformqueue()
    ----------------------------------------- #>
    [void]updatemainvminformqueue($currentprojecttasks, $cproject){
        #
        if (!$this.maincsv){
            $this.maincsv = @()
        }
        #
        $activetasks = $this.localqueue.($cproject) | & { process {
            if ($_.Algorithm.Trim() -ne '') { $_ }}}
        #
        if ($currentprojecttasks){
            #
            $differenttasks = Compare-Object -ReferenceObject $currentprojecttasks `
                -DifferenceObject $activetasks `
                -Property 'localtaskid','SlideID','Antibody','Algorithm' | 
                & { process { if ($_.SideIndicator -match '=>'){$_}}}
        } else {
            $differenttasks = $activetasks
        }
        #
        $differenttasks | & { process {
            #
            $this.maincsv += $_ | 
                select-object -Property slideid, Antibody, Algorithm |
                Add-Member -NotePropertyMembers @{
                    ProcessingLocation = ''
                    StartDate = ''
                    localtaskid = $_.localtaskid
                    TaskID = (
                        'T' + $cproject.PadLeft(3,'0') + $_.localtaskid.PadLeft(5,'0')
                    )
                } -PassThru
            #
        }}
        #
        $differenttasks = $null
        $activetasks = $null
        $currentprojecttasks = $null
        #
    }
    <# -----------------------------------------
     checkfornewtask
     checks if the supplied project, slideid, 
     and antibody present a new task for the 
     queue. If they do the local queue is updated.
     for a new row.
     ------------------------------------------
     Usage: $this.checkfornewtask()
    ----------------------------------------- #>
    [switch]checkfornewtask($project, $slideid, $antibody){
        #
        $this.getlocalqueue($project)
        $this.getlocalslide($slideid, $project)
        #
        if (!($this.localslide.antibody -contains $antibody)){
            $this.newlocalrow($project, $slideid, $antibody)
            $this.writelocalqueue($project)
            return $true
        } 
        #
        return $false
        #
    }
    #
    [void]getlocalslide($slideid, $cproject){
        #
        if (!($this.localslide.slideid -contains $slideid)){
            $this.localslide = $this.localqueue.($cproject) |
             & { process {   
                if (
                    $_.slideid -contains $slideid
                ) { $_ }}}
        }
        #
    }
    <# -----------------------------------------
     checkforreadytask
     checks if the supplied project, slideid, 
     and antibody present a task in the main
     queue that is ready to be run
     ------------------------------------------
     Usage: $this.checkforreadytask()
    ----------------------------------------- #>
    [string]checkforreadytask($project, $slideid, $antibody){
        #
        $this.getmainslide($slideid)
        #
        $task = $this.mainslide | & { process {   
            if (
                $_.Antibody -contains $antibody -and 
                $_.Algorithm -ne '' -and
                $_.ProcessingLocation -eq ''
            ) { $_ }}}
        #
        if ($task){
            return $task.taskid
        } 
        #
        return ''
        #
    }
    #
    [void]getmainslide($slideid){
        #
        if (!($this.mainslide.slideid -contains $slideid)){
            $this.mainslide = $this.maincsv |
             & { process {   
                if (
                    $_.slideid -contains $slideid
                ) { $_ }}}
        }
        #
    }
    <# -----------------------------------------
     checkforidletask
     checks if the supplied project, slideid, 
     and antibody present a task in the main
     queue that is ready to be run
     ------------------------------------------
     Usage: $this.checkforreadytask()
    ----------------------------------------- #>
    [switch]checkforidletask($project, $slideid, $antibody){
        #
        $this.getlocalqueue($project)
        $this.getlocalslide($slideid, $project)
        #
        if (
            $this.localslide | & { process {   
            if (
                $_.Antibody -contains $antibody -and 
                $_.Algorithm -eq ''
        ) {$_}}}){
            return $true
        } 
        #
        return $false
        #
    }
    #
    [void]newlocalrow($project, $slideid, $antibody){
        #
        if (!$this.localqueue.($project)){
            $this.localqueue.($project) = @()
        }
        #
        $newid = ($this.localqueue.($project).taskid |
            measure-object -maximum).Maximum + 1
        $this.localqueue.($project) += [PSCustomObject]@{
            TaskID = $newid
            localtaskid = $newid
            slideid = $slideid
            Antibody = $antibody
            Algorithm = ''
        }
        #
        $newid = $null
    }
    #
}
