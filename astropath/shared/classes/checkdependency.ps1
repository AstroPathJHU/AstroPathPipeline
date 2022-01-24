﻿<# -------------------------------------------
 sampletracker
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build the sample trackers for each
 sample and module
 -------------------------------------------#>
 class checkdependency {
    #
    checkdependency(){}
    #
    [string]getlogstatus([mylogger]$log, $cmodule){
            #
            if ($cmodule -match 'batch'){
                $log.slidelog = $log.mainlog
            }
            #
            if ($this.checklog($log, $false)){
                #
                $statusval = ($this.('check'+$cmodule)($log, $false))
                if ($statusval -eq 1){
                    $status = 'WAITING'
                } elseif ($statusval -eq 2){
                    $status = 'READY'
                } elseif ($statusval -eq 3){
                    $status = 'FINISHED'
                } else {
                    $status = 'UNKNOWN'
                }
                #
            } else {
                $status = 'RUNNING'
            }
            #
            return $status
            #
    }
    <# -----------------------------------------
     checklog
     check the provide log to see if the log
     exists and if the task has finished. 
     ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: 
     returns true if the slide has not yet 
     started or if there was an error between
     runs. if it is a dependency run it returns
     false if the task is finished and true if
     the task is still running (as inidicated by
     the logs). If is not a dependency it returns
     the opposite.
     ------------------------------------------
     Usage: $this.checklog(log, dependency)
    ----------------------------------------- #>
    [switch]checklog([mylogger]$log, $dependency){
        #
        if (!(test-path $log.slidelog)){
            return $true
        }
        #
        $loglines = $log.opencsvfile($log.slidelog, ';', @('Project','Cohort','slideid','Message','Date'))
        #
        # parse log
        #
        $statustypes = @('START:','ERROR:','FINISH:')
        $savelog = @()
        $vers = $log.vers -replace 'v', ''
        $vers = ($vers -split '\.')[0,1,2] -join '.'
        #
        if ($log.slidelog -match [regex]::Escape($log.mainlog)){
            $ID= $log.BatchID
        } else {
            $ID = $log.slideid
        }
        #
        foreach ($statustype in $statustypes){
            $savelog += $loglines |
                    where-object {
                        ($_.Message -match $vers) -and 
                         ($_.Slideid -match $ID) -and 
                         ($_.Message -match $statustype)
                    } |
                    Select-Object -Last 1 
        }
        #
        $d1 = ($savelog | Where-Object {$_.Message -match $statustypes[0]}).Date
        $d2 = ($loglines |
                 Where-Object {
                    $_.Message -match $statustypes[1] -and
                     ($_.Slideid -match $ID)
                 }).Date |
               Select-Object -Last 1 
        $d3 = ($savelog | Where-Object {$_.Message -match $statustypes[2]}).Date
        #
        # if there was an error return true 
        # if not a dependency check and the latest run is finished return true
        # if it is a dependency check and the lastest run is not finished return true
        #
        if ( !$d1 -or
             ($d1 -le $d2 -and $d3 -ge $d2) -or 
            (!$dependency -and ($d3 -gt $d1)) -or 
            ($dependency -and !($d3 -gt $d1))
        ){
            return $true
        } else {
            return $false
        }
        #
    }
    <# -----------------------------------------
     checktransfer
     check that the transfer process has completed
     and all transfer products exist
    ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checktransfer(log, dependency)
    ----------------------------------------- #>
    [int]checktransfer([mylogger]$log){
        #
        $log = [mylogger]::new($this.mpath, 'transfer', $log.slideid)
        #
        if (!($log.vers -match '0.0.1') -and 
            $this.checklog($log, $true)){
            return 2
        }
        #
        # check for checksum, qptiff, and annotationxml
        #
        $file = $log.CheckSumsfile()
        $file2 = $log.qptifffile()
        $file3 = $log.annotationxml()
        $im3s = (gci ($log.Scanfolder() + '\MSI\*') *im3).Count
        #
        #if (!(test-path $file)){
        #    return 2
        #}
        if (!(test-path $file2)){
            return 2
        }
        if (!(test-path $file3)){
            return 2
        }    
        if (!$im3s){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkshredxml
     check that the shredxml module has completed
     and all products exist
    ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkshredxml(log, dependency)
    ----------------------------------------- #>
    [int]checkshredxml([mylogger]$log, $dependency){
        #
        if (!($this.checktransfer($log) -eq 3)){
            return 1
        }
        #
        $log = [mylogger]::new($this.mpath, 'shredxml', $log.slideid)
        if ($this.checklog($log, $true)){
            return 2
        }
        #
        if (!$log.testxmlfiles()){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkmeanimage
     check that the meanimage module has completed
     and all products exist
    ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkmeanimage(log, dependency)
    ----------------------------------------- #>
    [int]checkmeanimage([mylogger]$log, $dependency){
        #
        if (!($this.checkshredxml($log, $true) -eq 3)){
            return 1
        }
        #
        $log = [mylogger]::new($this.mpath, 'meanimage', $log.slideid)
        if ($this.checklog($log, $true)){
            return 2
        }
        #
        if (!$log.testmeanimagefiles()){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkmeanimagecomparison
     check that the meanimagecomparison module has completed
     and all products exist
    ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkmeanimagecomparison(log, dependency)
    ----------------------------------------- #>
    [int]checkbatchmicomp([mylogger]$log, $dependency){
        #
        # if task is not a dependency and the version is
        # 0.0.1 then just checkout
        #
        if (
            !$dependency -and
             $log.vers -match '0.0.1'
            ){
            return 1
        }
        #
        if (!($this.checkmeanimage($log, $true) -eq 3)){
            return 1
        }
        #
        $log = [mylogger]::new($this.mpath, 'batchmicomp', $log.slideid)
        $log.slidelog = $log.mainlog
        if ($this.checklog($log, $true)){
            return 2
        }
        #
        # get the meanimagecomparison table  
        # extract current dpath from root_dir_1
        # check if slideID is in slideid 1
        # do the same on root 2
        # if slide not yet then return 2
        #
        # if (!$log.testmeanimagecomparison()){
        #    return 2
        #}
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkbatchflatfield
     check that the batchflatfield module has completed
     and all products exist
    ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkbatchflatfield(log, dependency)
    ----------------------------------------- #>
    [int]checkbatchflatfield([mylogger]$log, $dependency){
        #
        # if task is not a dependency and the version is
        # not 0.0.1 then just checkout
        #
        if (
            !$dependency -and
             $log.vers -notmatch '0.0.1'
            ){
            return 1
        }
        #
        $log = [mylogger]::new($this.mpath, 'batchflatfield', $log.slideid)
        #
        # if the version is not 0.0.1 in batchflatfield, do meanimagecomparison
        # instead
        if ($log.vers -notmatch '0.0.1'){
            #
            if (!($this.checkbatchmicomp($log, $true) -eq 3)){
                return 1
            }
            #
            $ids = $this.ImportCorrectionModels($this.mpath)
            if ($ids.slideid -notcontains $log.slideid){
                return 2
            }
            #
            if (!$log.testpybatchflatfield()){
                return 2
            }
            #
        } else {
            #
            if (!($this.checkmeanimage($log, $true) -eq 3)){
                return 1
            }
            #
            $log.slidelog = $log.mainlog
            if ($this.checklog($log, $true)){
                return 2
            }
            #
            if (!$log.testbatchflatfield()){
                return 2
            }
            #
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkwarpoctets
     check that the meanimage module has completed
     and all products exist
    ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkmeanimage(log, dependency)
    ----------------------------------------- #>
    [int]checkwarpoctets([mylogger]$log, $dependency){
        #
        if (!($this.checkbatchflatfield($log, $true) -eq 3)){
            return 1
        }
        #
        $log = [mylogger]::new($this.mpath, 'warpoctets', $log.slideid)
        if ($this.checklog($log, $true)){
            return 2
        }
        #
        if (!$log.testwarpoctets()){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkimagecorrection
     check that the imagecorrection module has completed
     and all products exist
    ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkimagecorrection(log, dependency)
    ----------------------------------------- #>
    [int]checkimagecorrection([mylogger]$log, $dependency){
        #
        if (!($this.checkbatchflatfield($log, $true) -eq 3)){
            return 1
        }
        #
        $log = [mylogger]::new($this.mpath, 'imagecorrection', $log.slideid)
        if ($this.checklog($log, $true)){
            return 2
        }
        #
        if(!$log.testimagecorrectionfiles()){
            return 2
        }
        #
        return 3
    }
    <# -----------------------------------------
     checkvminform
     check that the vminform module has completed
     and all products exist
    ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkvminform(log, dependency)
    ----------------------------------------- #>
    [void]checkvminform(){
        #
        $this.informvers = '2.6.0'
        #
        $queue_path = $this.mpath + '\across_project_queues'
        $this.queue_file = $queue_path + '\' + $this.module + '-queue.csv'
        $queue_data = $this.getcontent($this.queue_file)
        #
        $current_queue_data = @()
        #
        # find rows without "processing started"
        #
        foreach($row in $queue_data) {
            $array = $row.ToString().Split(",")
            $array = $array -replace '\s',''
            if($array[3]){
                if($array -match "Processing"){ Continue } else { 
                    $current_queue_data += $row
                    }
                } 
        }
        #
        $this.originaltasks = $current_queue_data
        $this.cleanedtasks = $this.originaltasks -replace ('\s','')
        $this.cleanedtasks = $this.cleanedtasks | ForEach {$_.Split(',')[0..3] -join(',')}
        #
    }
    <# -----------------------------------------
     Aggregatebatches
     check that all slides from each unqiue batch are on the list
     return one sample
    ------------------------------------------
     Input: 
        - batcharry[array]: project, batch pairs
            for each slide that has finished.
     ------------------------------------------
     Output: returns a list of unique project batch 
     pairs that have all slides complete
     ------------------------------------------
     Usage: $this.Aggregatebatches(batcharray)
    ----------------------------------------- #>
    [array]Aggregatebatches($batcharray, $cmodule){
        $batcharrayunique = $batcharray | Sort-Object | Get-Unique
        $slides = $this.importslideids($this.mpath)
        $batchescomplete = @()
        #
        $batcharrayunique | foreach-object {
            $nslidescomplete = ($batcharray -match $_).count
            $projectbatchpair = $_ -split ','
            $sample = [sampledef]::new($this.mpath, $cmodule, $projectbatchpair[1], $projectbatchpair[0])
            $nslidesbatch = $sample.batchslides.count
            if ($nslidescomplete -eq $nslidesbatch){
                $batchescomplete += $_
            }
        }
        return $batchescomplete
    }
    #
}