﻿<# -------------------------------------------
 vminformqueue
 created by: Benjamin Green, Andrew Joqurea - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 a special case of the queue functionality
 -------------------------------------------#>
class vminformqueue : sharedtools {
    #
    [PSCustomObject]$mainvminformcsv
    [hashtable]$localvminformdates = @{}
    [hashtable]$localqueue = @{}
    [hashtable]$localqueuefile = @{}
    [string]$localqueue_file = 'inForm_queue.csv'
    [string]$mainqueue_file ='vminform-queue.csv'
    [string]$mainqueue_path = '\across_project_queues'
    #
    vminformqueue(){}
    #
   # vminformqueue(){
   #     $this.mpath = '\\bki04\astropath_processing'
   # }
    vminformqueue($mpath){
        $this.mpath = $mpath
    }
    vminformqueue($mpath, $project){
        $this.mpath = $mpath
        $this.project = $project
    }
  <# -----------------------------------------
     coalescevminformqueues
     coalesce local and main vminform queues
     ------------------------------------------
     Usage: $this.coalescevminformqueues()
    ----------------------------------------- #>
    [void]coalescevminformqueues(){
         $projects = $this.getapprojects('vminform')
         #
         $projects | ForEach-Object{
            $this.coalescevminformqueues($_)
         }
         #
    }
  <# -----------------------------------------
     coalescevminformqueues
     coalesce local and main vminform queues
     for a particular project
     ------------------------------------------
     Usage: $this.coalescevminformqueues($project)
    ----------------------------------------- #>
    [void]coalescevminformqueues($project){
        #
        $project = $project.ToString()
        $this.openmainvminformqueue()
        $currentprojecttasks = $this.mainvminformcsv -match ('T' + $project.PadLeft(3,'0'))
        #
        $this.getlocalvminformqueue($project)
        #
        $this.updatelocalvminfomqueue($currentprojecttasks, $project)
        $this.mainvminformcsv = $this.updatemainvminfomqueue($project, $this.mainvminformcsv)
        #
        $this.writelocalqueue($project)
        $this.writemainqueue($this.mainvminformcsv, $this.mainvminformqueuelocation())
        #
    }
    <# -----------------------------------------
     mainvminfomqueuelocation
     open main inform queue
     ------------------------------------------
     Usage: $this.mainvminfomqueuelocation()
    ----------------------------------------- #>
    [string]mainvminformqueuelocation(){
        #
        $mainqueuefile = $this.mpath +
             $this.mainqueue_path + '\' + $this.mainqueue_file
        #
        return $mainqueuefile
        #
    } 
    #
    [void]openmainvminformqueue(){
       $this.openmainvminformqueue($false)
    }

    <# -----------------------------------------
     openmainvminfomqueue
     open main inform queue
     ------------------------------------------
     Usage: $this.openmainvminfomqueue()
    ----------------------------------------- #>
    [void]openmainvminformqueue($createwatcher){
        #
        $mainqueuefile = $this.mainvminformqueuelocation()
        #
        if (!(test-path $mainqueuefile)){
            $this.setfile($mainqueuefile,
                'TaskID,Specimen,Antibody,Algorithm,ProcessingLocation,StartDate')
        }
        #
        $mainqueue = $this.OpenCSVFile($mainqueuefile)
        $mainqueue = $mainqueue | where-object { $_.taskid.Length -eq 9}
        if ($mainqueue){
            $mainqueue | foreach-object {
                $_ | Add-Member localtaskid $_.taskid.substring(4).trim('0') -PassThru
            }
        }
        #
        if ($createwatcher){
            $this.FileWatcher($mainqueuefile)
        }
        #
        $this.mainvminformcsv = $mainqueue
    }
    <# -----------------------------------------
     openmainvminfomqueue
     open main inform queue
     ------------------------------------------
     Usage: $this.openmainvminfomqueue()
    ----------------------------------------- #>
    [void]openlocalvminformqueue($project){
        #
        if (!($this.localqueuefile.($project))){
            $this.openlocalvminformqueue($project, $false)
        }
        #
    }
    #
    [void]openlocalvminformqueue($project, $createwatcher){
        #
        if (!(test-path $this.localqueuefile.($project))){
            $this.setfile($this.localqueuefile.($project), 'TaskID,Specimen,Antibody,Algorithm')
        }
        #
        if ($createwatcher){
            $this.FileWatcher($this.localqueuefile.($project))
        }
        #
        $this.localqueue.($project) = $this.OpenCSVFile($this.localqueuefile.($project))
        #
        if (!$this.localqueue.($project)){
            $this.localqueue.($project) = @()
        }
        #
    }  
    <# -----------------------------------------
     localvminfomqueuelocation
     open local inform queue
     ------------------------------------------
     Usage: $this.localvminfomqueuelocation()
    ----------------------------------------- #>
    [string]localvminfomqueuelocation($project){
        #
        $cohortinfo = $this.GetProjectCohortInfo($this.mpath, $project)
        $localqueuepath = $cohortinfo.Dpath +
            '\' + $cohortinfo.Dname + '\upkeep_and_progress'
        #
        if ($this.isWindows()){
            $localqueuepath = '\\' + $localqueuepath
        }
        #
        $localqueuefilea = $localqueuepath + '\' + $this.localqueue_file
        #
        return $localqueuefilea
        #
    }
    #
    [void]getlocalvminformqueue($project){
        #
        $this.localqueuefile.($project) = $this.localvminfomqueuelocation($project)
        $this.openlocalvminformqueue($project)
        #
    }
    #
    [void]getlocalvminformqueue($project, $createwatcher){
        #
        $this.localqueuefile.($project) = $this.localvminfomqueuelocation($project)
        $this.openlocalvminformqueue($project, $createwatcher)
        #
    }
    <# -----------------------------------------
     updatelocalvminfomqueue
     update local vminform queue
     ------------------------------------------
     Usage: $this.updatelocalvminfomqueue()
    ----------------------------------------- #>
    [void]updatelocalvminfomqueue($currentprojecttasks, $project){
        #
        $currentprojecttasks | foreach-object {
            $localtaskid = $_.localtaskid
            #
            # tasks in main not in local
            #
            if ($localtaskid -notin $this.localqueue.($project).TaskID) {
                $NewRow =  $_ |
                     select-object -Property Specimen, Antibody, Algorithm |
                     Add-Member TaskID $localtaskid -PassThru
                $localqueue += $NewRow
            }
            #
            # if local row does not match main, move local row to next taskid and 
            # replace old local row with the row in main
            #
            $localrow = $this.localqueue.($project) | Where-Object -FilterScript {$_.TaskID -eq $row.TaskID}
            if ($localrow -notmatch ($row |
                 select-object -Property TaskID, Specimen, Antibody, Algorithm)) {
                $newid = ($this.localqueue.($project).taskid | measure-object -maximum).Maximum + 1
                $NewRow =  $localrow |
                select-object -Property Specimen, Antibody, Algorithm |
                    Add-Member TaskID $newid -PassThru
                $this.localqueue.($project) += $NewRow
                #
                $localrow.Specimen = $_.Specimen
                $localrow.Antibody = $_.Antibody
                $localrow.Algorithm = $_.Algorithm
            }
        }
        #
    }
    <# -----------------------------------------
     updatemainvminfomqueue
     update main vminform queue
     ------------------------------------------
     Usage: $this.updatemainvminfomqueue()
    ----------------------------------------- #>
    [PSCustomObject]updatemainvminfomqueue($project, $mainqueue){
        #
        $this.localqueue.($project) | ForEach-Object{
            if ($_.TaskID -notin $mainqueue.localtaskid -and $_.Algorithm -ne '') {
                
                $NewRow = $_ | 
                    select-object -Property TaskID, Specimen, Antibody, Algorithm |
                    Add-Member ProcessingLocation, StartDate '','' -PassThru
                $NewRow.TaskID = 'T' + $project.PadLeft(3,'0') + ($NewRow.TaskID.ToString()).PadLeft(5,'0')
                $mainqueue += $NewRow
            }
        }
        #
        return $mainqueue
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
        $this.getlocalvminformqueue($project, $false)
        #
        $task = $this.localqueue.($project) |    
            Where-Object {$_.Specimen -match $slideid -and 
                $_.Anitbody -match $antibody    
            
            }
        #
        if (!$task){
            $this.newlocalrow($this.localqueue.($project), $slideid, $antibody)
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
        $task = $this.mainvminformcsv|    
            Where-Object {$_.Specimen -match $slideid -and 
                $_.Anitbody -match $antibody -and 
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
        $task = $this.mainvminformcsv|    
            Where-Object {$_.Specimen -match $slideid -and 
                $_.Anitbody -match $antibody -and 
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
        $newid = ($this.localqueue.($project).taskid | measure-object -maximum).Maximum + 1
        $NewRow =  [PSCustomObject]@{
            TaskID = $newid
            Specimen = $slideid
            Antibody = $antibody
            Algorithm = ''
        } 
        $this.localqueue.($project) += $NewRow
        #
    }
    #
    [void]writelocalqueue($project){
        $updatedlocal = (($this.localqueue.($project)| 
            ConvertTo-Csv -NoTypeInformation) -join "`r`n").Replace('"','') + "`r`n"
        $this.SetFile($this.localqueuefile.($project), $updatedlocal)
    }
    #
    [void]writemainqueue($mainqueue, $mainqueuelocation){
        $mainqueue = $mainqueue | 
            select-object -Property TaskID, Specimen, Antibody, Algorithm, ProcessingLocation, StartDate
        $updatedmain = (($mainqueue | ConvertTo-Csv -NoTypeInformation) -join "`r`n").Replace('"','') + "`r`n"
        $this.SetFile($mainqueuelocation, $updatedmain)
    }
    #
    [void]UpdateQueue($currenttask, $currentworker, $tasktomatch){
        #
        if ($this.module -ne 'vminform'){
            return
        }
        #
        $D = Get-Date
        $currenttask2 = "$currenttask" + ",Processing: " + 
            $currentworker.server + '-' + $currentworker.location + "," + $D
        $mxtstring = 'Global\' + $this.queue_file.replace('\', '_') + '.LOCK'
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
                    $Q = get-content -Path $this.queue_file
                    $Q2 = $Q -replace $rg,$currenttask2
                    Set-Content -Path $this.queue_file -Value $Q2
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
