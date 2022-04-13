<# -------------------------------------------
 vminformqueue
 created by: Benjamin Green, Andrew Joqurea - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 a special case of the queue functionality
 -------------------------------------------#>
class vminformqueue : modulequeue {
    #
    [string]$informvers = '2.4.8'
    #
    vminformqueue() : base ('vminform'){
        $this.vminformqueueinit()
        #$this.coalescevminformqueues()
    }
    vminformqueue($mpath): base ($mpath, 'vminform'){
        $this.vminformqueueinit()
        #$this.coalescevminformqueues()
    }
    vminformqueue($mpath, $project) : base($mpath, 'vminform', $project){
        $this.vminformqueueinit()
        #$this.coalescevminformqueues($project)
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
        $projects = $this.getapprojects($this.module)
        #
        $this.openmainqueue($false)
        $projects | ForEach-Object{
            $this.coalescevminformqueues($_, $true)
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
        $localtmp = $this.localqueue.($cproject)
        $this.pairqueues($cproject)
        #
        if ($localtmp){
            $cmp = Compare-Object -ReferenceObject $localtmp.taskid `
                -DifferenceObject $this.localqueue.($cproject).taskid
            if ($cmp){
                $this.writelocalqueue($cproject)
            }
         } else {
            $this.writelocalqueue($cproject)
        }
        #
    }
    <#------------------------------------------
    will force update all local and main inform 
    queues and create file watchers. to run after the
    sampledb has been built
    --------------------------------------------#>
    [void]createwatchersvminformqueues(){
        #
        $projects = $this.getapprojects($this.module)
            #
        $this.openmainqueue($false)
        $projects | ForEach-Object{
            #
            $this.coalescevminformqueues($_, $false)
            $this.getlocalqueue($_, $true)
            #
        }
        #
        $this.writemainqueue($this.mainqueuelocation())
        $this.openmainqueue($true)
        #
    }
    #
    [void]pairqueues($cproject){
        #
        $currentprojecttasks = $this.filtertasks($cproject)
        $this.updatelocalvminformqueue($currentprojecttasks, $cproject)
        $this.updatemainvminformqueue($cproject)
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
        foreach ($row in $currentprojecttasks){
            $localtaskid = $row.localtaskid
            #
            # tasks in main not in local
            #
            if ($localtaskid -notin $this.localqueue.($project).TaskID) {
                $NewRow =  $row |
                     select-object -Property slideid, Antibody, Algorithm |
                     Add-Member TaskID $localtaskid -PassThru
               $this.localqueue.($project) += $NewRow
            }
            #
            # if local row does not match main, move local row to next taskid and 
            # replace old local row with the row in main
            #
            $localrow = $this.localqueue.($project) |
                Where-Object -FilterScript {$_.TaskID -eq $localtaskid}
            #
            while ($localrow.count -gt 1){
                $localrow[1].TaskID = ($this.localqueue.($project).taskid |
                    measure-object -maximum).Maximum + 1
                $localrow = $this.localqueue.($project) |
                    Where-Object -FilterScript {$_.TaskID -eq $localtaskid}
            }
            #
            if (($localrow |
                    select-object -Property slideid, Antibody, Algorithm) `
                -notmatch ($row |
                    select-object -Property slideid, Antibody, Algorithm)
            ) {
                #
                $max = ($this.localqueue.($project).taskid | measure-object -maximum).Maximum
                $newid = $max + 1
                $NewRow =  $localrow |
                select-object -Property slideid, Antibody, Algorithm |
                    Add-Member TaskID $newid -PassThru
                $this.localqueue.($project) += $NewRow
                #
                $localrow[0].slideid = $row.slideid
                $localrow[0].Antibody = $row.Antibody
                $localrow[0].Algorithm = $row.Algorithm
            }
        }
        #
    }
    <# -----------------------------------------
     updatemainvminformqueue
     update main vminform queue
     ------------------------------------------
     Usage: $this.updatemainvminformqueue()
    ----------------------------------------- #>
    [void]updatemainvminformqueue($project){
        #
        if (!$this.maincsv){
            $this.maincsv = @()
        }
        #
        $this.localqueue.($project) | ForEach-Object{
            #
            if ($_.Algorithm.Trim() -ne ''){
                #
                $maintaskid = 'T', $project.PadLeft(3,'0'),
                    ($_.TaskID.ToString()).PadLeft(5,'0') -join ''
                #
                if ($maintaskid -notin $this.maincsv.taskid) {
                    $NewRow = $_ | 
                        select-object -Property TaskID, slideid, Antibody, Algorithm |
                        Add-Member -NotePropertyMembers @{
                            ProcessingLocation = ''
                            StartDate = ''
                            localtaskid = $_.TaskID
                        } -PassThru
                    $NewRow.TaskID = $maintaskid
                    $this.maincsv += $NewRow
                }
            }
        }
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
        $this.getlocalqueue($project, $false)
        #
        $task = $this.localqueue.($project) |    
            Where-Object {
                $_.slideid -match $slideid -and 
                $_.Antibody -match $antibody    
            }
        #
        if (!$task){
            $this.newlocalrow($project, $slideid, $antibody)
            $this.writelocalqueue($project)
            return $true
        } 
        #
        return $false
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
    [switch]checkforreadytask($project, $slideid, $antibody){
        #
        $task = $this.maincsv |    
            Where-Object {$_.slideid -match $slideid -and 
                $_.Antibody -match $antibody -and 
                $_.Algorithm -ne '' -and
                $_.ProcessingLocation -eq ''
            
            }
        #
        if ($task){
            return $true
        } 
        #
        return $false
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
        $this.getlocalqueue($project, $false)
        #
        $task = $this.localqueue.($project) |    
            Where-Object {$_.slideid -match $slideid -and 
                $_.Antibody -match $antibody -and 
                $_.Algorithm -eq '' -and
                $_.ProcessingLocation -eq ''
            
            }
        #
        if ($task){
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
        $newid = ($this.localqueue.($project).taskid | measure-object -maximum).Maximum + 1
        $NewRow =  [PSCustomObject]@{
            TaskID = $newid
            slideid = $slideid
            Antibody = $antibody
            Algorithm = ''
        } 
        $this.localqueue.($project) += $NewRow
        #
    }
    #
    [void]UpdateQueue($currenttask, $currentworker, $tasktomatch){
        #
        if ($this.module -ne $this.module){
            return
        }
        #
        $D = Get-Date
        $currenttask2 = "$currenttask" + ",Processing: " + 
            $currentworker.server + '-' + $currentworker.location + "," + $D
        $mxtstring = 'Global\' + ($this.mainqueuelocation()).replace('\', '_') + '.LOCK'
        #
        # add escape to '\'
        #
        $rg = [regex]::escape($tasktomatch) + "$"
        #
        $cnt = 0
        $Max = 120
        #
        do{
           $mxtx = New-Object System.Threading.Mutex($false, $mxtstring)
            try{
                $imxtx = $mxtx.WaitOne(60 * 10)
                if($imxtx){
                    $Q = get-content -Path $this.mainqueuelocation()
                    $Q2 = $Q -replace $rg,$currenttask2
                    Set-Content -Path $this.mainqueuelocation() -Value $Q2
                    $mxtx.releasemutex()
                    break
                } else{
                    $cnt = $cnt + 1
                    Start-Sleep -s 5
                }
            }catch{
                $cnt = $cnt + 1
                Start-Sleep -s 5
                Continue
            }
        } while($cnt -lt $Max)
        #
        # if the script could not access the queue file after 10 mins of trying every 2 secs
        # there is an issue and exit the script
        #
        if ($cnt -ge $Max){
            $ErrorMessage = "Could not access "+$this.module+"-queue.csv"
            Throw $ErrorMessage 
        }
        #
    }
}
