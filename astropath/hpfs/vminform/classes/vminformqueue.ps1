<# -------------------------------------------
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
        $mainqueue = $this.openmainvminformqueue()
        $currentprojecttasks = $mainqueue -match ('T' + $project.PadLeft(3,'0'))
        #
        $localqueuefile = $this.localvminfomqueuelocation($project)
        $this.localvminformdates.($project) = $this.LastWrite($localqueuefile)
        $localqueue = $this.openlocalvminformqueue($localqueuefile)
        #
        $localqueue = $this.updatelocalvminfomqueue($currentprojecttasks, $localqueue)
        $mainqueue = $this.updatemainvminfomqueue($project, $mainqueue, $localqueue)
        #
        $this.writelocalqueue($localqueue, $localqueuefile)
        $this.writemainqueue($mainqueue, $this.mainvminformqueuelocation())
        $this.mainvminformcsv = $mainqueue
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
        $mainqueuepath = $this.mpath + '\across_project_queues'
        $mainqueuefile = $mainqueuepath + '\vminform-queue.csv'
        #
        return $mainqueuefile
        #
    } 
    <# -----------------------------------------
     openmainvminfomqueue
     open main inform queue
     ------------------------------------------
     Usage: $this.openmainvminfomqueue()
    ----------------------------------------- #>
    [PSCustomObject]openmainvminformqueue(){
        #
        $mainqueuefile = $this.mainvminformqueuelocation()
        #
        if (!(test-path $mainqueuefile)){
            $this.setfile($mainqueuefile, 'TaskID,Specimen,Antibody,Algorithm,ProcessingLocation,StartDate')
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
        return $mainqueue
        #
    }
    <# -----------------------------------------
     openmainvminfomqueue
     open main inform queue
     ------------------------------------------
     Usage: $this.openmainvminfomqueue()
    ----------------------------------------- #>
    [PSCustomObject]openlocalvminformqueue($localqueuefile){
        #
        if (!(test-path $localqueuefile)){
            $this.setfile($localqueuefile, 'TaskID,Specimen,Antibody,Algorithm')
        }
        #
        $localqueue = $this.OpenCSVFile($localqueuefile)
        #
        if (!$localqueue){
            $localqueue = @()
        }
        #
        return $localqueue
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
        $localqueuepath = $cohortinfo.Dpath + '\' + $cohortinfo.Dname + '\upkeep_and_progress'
        #
        if ($this.isWindows()){
            $localqueuepath = '\\' + $localqueuepath
        }
        #
        $localqueuefile = $localqueuepath + '\inForm_queue.csv' 
        #
        return $localqueuefile
        #
    }
    <# -----------------------------------------
     updatelocalvminfomqueue
     update local vminform queue
     ------------------------------------------
     Usage: $this.updatelocalvminfomqueue()
    ----------------------------------------- #>
    [PSCustomObject]updatelocalvminfomqueue($currentprojecttasks, $localqueue){
        #
        $currentprojecttasks | foreach-object {
            $localtaskid = $_.localtaskid
            #
            # tasks in main not in local
            #
            if ($localtaskid -notin $localqueue.TaskID) {
                $NewRow =  $_ |
                     select -Property Specimen, Antibody, Algorithm |
                     Add-Member TaskID $localtaskid -PassThru
                $localqueue += $NewRow
            }
            #
            # if local row does not match main, move local row to next taskid and 
            # replace old local row with the row in main
            #
            $localrow = $localqueue | Where-Object -FilterScript {$_.TaskID -eq $row.TaskID}
            if ($localrow -notmatch ($row | select -Property TaskID, Specimen, Antibody, Algorithm)) {
                $addedrow = New-Object System.Object
                $newid = ($localqueue.taskid | measure -maximum).Maximum + 1
                $NewRow =  $localrow |
                    select -Property Specimen, Antibody, Algorithm |
                    Add-Member TaskID $newid -PassThru
                $localqueue += $NewRow
                #
                $localrow.Specimen = $_.Specimen
                $localrow.Antibody = $_.Antibody
                $localrow.Algorithm = $_.Algorithm
            }
        }
        #
        return $localqueue
    }
    <# -----------------------------------------
     updatemainvminfomqueue
     update main vminform queue
     ------------------------------------------
     Usage: $this.updatemainvminfomqueue()
    ----------------------------------------- #>
    [PSCustomObject]updatemainvminfomqueue($project, $mainqueue, $localqueue){
        #
        $localqueue | ForEach-Object{
            if ($_.TaskID -notin $mainqueue.localtaskid -and $_.Algorithm -ne '') {
                
                $NewRow = $_ | select -Property TaskID, Specimen, Antibody, Algorithm |
                    Add-Member ProcessingLocation, StartDate '','' -PassThru
                $NewRow.TaskID = 'T' + $project.PadLeft(3,'0') + ($NewRow.TaskID.ToString()).PadLeft(5,'0')
                $mainqueue += $NewRow
            }
        }
        #
        return $mainqueue
    }
    #
    [void]writelocalqueue($localqueue, $localqueuelocation){
        $updatedlocal = (($localqueue | ConvertTo-Csv -NoTypeInformation) -join "`r`n").Replace('"','') + "`r`n"
        $this.SetFile($localqueuelocation, $updatedlocal)
    }
    #
    [void]writemainqueue($mainqueue, $mainqueuelocation){
        $mainqueue = $mainqueue | select -Property TaskID, Specimen, Antibody, Algorithm, ProcessingLocation, StartDate
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
        $currenttask2 = "$currenttask" + ",Processing: " + $currentworker.server + '-' + $currentworker.location + "," + $D
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
