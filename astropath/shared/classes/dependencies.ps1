﻿<# -------------------------------------------
 dependencies
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build the sample trackers for each
 sample and module
 -------------------------------------------#>
 class dependencies : samplereqs {
    #
    [hashtable]$status = 
     @{finish = 'FINISHED';
        running = 'RUNNING';
        ready = 'READY';
        error = 'ERROR';
        waiting = 'WAITING';
        na = 'NA';
        unknown = 'UNKNOWN';
        rerun = 'RERUN'
    }
    #
    [string]$empty_time = 'NA'
    #
    $pipeline_steps = [ordered]@{
        step1 = 'scan';
        step2 = 'transfer';
        step3 = 'shredxml';
        step4 = 'meanimage';
        step5 = 'batchmicomp';
        step6 = 'batchflatfield';
        step7 = 'warpoctets';
        step8 = 'batchwarpkeys';
        step9 = 'batchwarpfits';
        step10 = 'imagecorrection';
        step11 = 'vminform';
        step12 = 'merge';
        step13 = 'imageqa';
        step14 = 'segmaps';
        step15 = 'dbload'
    }
    #
    dependencies($mpath): base ($mpath){}
    #
    dependencies($mpath, $slideid): base ($mpath, '', $slideid){}
    #
    # dependencies($mpath, $module, $batchid, $project) : base ($mpath, $module, $batchid, $project){}
    #
    [void]getlogstatus($cmodule){
        #
        if ($cmodule -match 'vminform'){
            #
            $this.getantibodies()
            foreach ($abx in $this.antibodies) {
                $this.getlogstatussub($cmodule, $abx)
            }
            #
        } else {
            $this.getlogstatussub($cmodule)
        }
        #
    }
    #
    [void]getantibodies(){
        try{
            $this.findantibodies($this.basepath)
        } catch {
            Write-Host $_.Exception.Message
            return
        }
    }
    #
    [void]getlogstatussub($cmodule){
        #
        $logoutput = $this.checkloginit($cmodule, $false)
        #
        if ($logoutput[1]){
            $this.moduleinfo.($cmodule).status = $logoutput[1].Message
        } elseif ($logoutput) {
            #
            switch ($this.('check'+$cmodule)()){
                1 {$this.moduleinfo.($cmodule).status = $this.status.waiting}
                2 {$this.moduleinfo.($cmodule).status = $this.status.ready}
                3 {$this.moduleinfo.($cmodule).status = $this.status.finish}
                4 {$this.moduleinfo.($cmodule).status = $this.status.na}
                Default {$this.moduleinfo.($cmodule).status = $this.status.unknown}
            }
            #
        } else {
            #
            $this.moduleinfo.($cmodule).status = $this.status.running
            #
        }
        #
    }
    #
    [void]getlogstatussub($cmodule, $antibody){
        #
        $this.moduleinfo.($cmodule).($antibody) = @{}
        $logoutput = $this.checkloginit($cmodule, $antibody, $false)
        <#
        if ($logoutput[1]){
            $this.moduleinfo.($cmodule).($antibody).status = $logoutput[1].Message
        } elseif ($logoutput) {
            #>
        if ($logoutput){
            switch ($this.('check'+$cmodule)($antibody)){
                1 {$this.moduleinfo.($cmodule).($antibody).status = $this.status.waiting}
                2 {$this.moduleinfo.($cmodule).($antibody).status = $this.status.ready}
                3 {$this.moduleinfo.($cmodule).($antibody).status = $this.status.finish}
                4 {$this.moduleinfo.($cmodule).($antibody).status = $this.status.na}
                5 {$this.moduleinfo.($cmodule).($antibody).status = $logoutput[1].Message}
                Default {$this.moduleinfo.($cmodule).($antibody).status = $this.status.unknown}
            }
            #
        } else {
            #
            $this.moduleinfo.($cmodule).($antibody).status = $this.status.running
            #
        }
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
    [array]checklog($cmodule, $dependency){
        #
        if (!($this.modulelogs.($cmodule).($this.project))){
            return @($true)
        }
        #
        $startdate = $this.moduleinfo.($cmodule).StartTime
        $finishdate = $this.moduleinfo.($cmodule).FinishTime
        $errorline = $this.moduleinfo.($cmodule).Errorline
        #
        return ($this.deflogstatus($startdate, $finishdate, $errorline, $dependency))
        #
    }
    #
    [array]checkloginit($cmodule, $dependency){
        #
        if (!($this.modulelogs.($cmodule).($this.project))){
            $this.moduleinfo.($cmodule).StartTime = $this.empty_time
            $this.moduleinfo.($cmodule).FinishTime = $this.empty_time
            return @($true)
        }
        #
        $loglines = $this.modulelogs.($cmodule).($this.project)
        $vers = $this.setlogvers($cmodule)
        $ID = $this.setlogid($cmodule)
        #
        $startdate = ($this.selectlogline($loglines,
            $ID, $this.log_start, $vers)).Date
        $finishdate = ($this.selectlogline($loglines,
            $ID, $this.log_finish, $vers)).Date
        $errorline = $this.selectlogline($loglines,
            $ID, $this.log_error)
        #
        if ($startdate){
            $this.moduleinfo.($cmodule).StartTime = $startdate
        } else {
            $this.moduleinfo.($cmodule).StartTime = $this.empty_time
        }
        #
        if ($startdate -and $finishdate -and 
                (get-date $finishdate) -gt (get-date $startdate)
        ){
            $this.moduleinfo.($cmodule).FinishTime = $finishdate
        } else {
            $this.moduleinfo.($cmodule).FinishTime = $this.empty_time
        }
        #
        $this.moduleinfo.($cmodule).Errorline = $errorline
        #
        return ($this.deflogstatus($startdate, $finishdate, $errorline, $dependency))
        #
    }
    #
    [array]checklog($cmodule, $antibody, $dependency){
        #
        if (!($this.modulelogs.($cmodule).($this.project))){
            return @($true)
        }
        #
        $startdate = $this.moduleinfo.($cmodule).($antibody).StartTime
        $finishdate = $this.moduleinfo.($cmodule).($antibody).FinishTime
        $errorline = $this.moduleinfo.($cmodule).($antibody).Errorline
        #
        return ($this.deflogstatus($startdate, $finishdate, $errorline, $dependency))
        #
    }
    #
    [array]checkloginit($cmodule, $antibody, $dependency){
        #
        if (!($this.modulelogs.($cmodule).($this.project))){
            $this.moduleinfo.($cmodule).($antibody).StartTime = $this.empty_time
            $this.moduleinfo.($cmodule).($antibody).FinishTime = $this.empty_time
            return @($true)
        }
        #
        $loglines = $this.modulelogs.($cmodule).($this.project)
        $vers = $this.setlogvers($cmodule)
        $ID = $this.setlogid($cmodule)
        #
        $startdate = ($this.selectlogline($loglines,
            $ID, $this.log_start, $vers, $antibody)).Date
        $finishdate = ($this.selectlogline($loglines,
         $ID, $this.log_finish, $vers, $antibody)).Date
        $errorline = $this.selectlogline($loglines,
         $ID, $this.log_error, '', $antibody)
        #
        if ($startdate){
            $this.moduleinfo.($cmodule).($antibody).StartTime = $startdate
        } else {
            $this.moduleinfo.($cmodule).($antibody).StartTime = $this.empty_time
        }
        #
        if ($startdate -and $finishdate -and
            (get-date $finishdate) -gt (get-date $startdate)){
            $this.moduleinfo.($cmodule).($antibody).FinishTime = $finishdate
        } else {
            $this.moduleinfo.($cmodule).($antibody).FinishTime = $this.empty_time
        }
        #
        $this.moduleinfo.($cmodule).($antibody).Errorline = $errorline
        #
        return ($this.deflogstatus($startdate, $finishdate, $errorline, $dependency))
        #
    }
    <# -----------------------------------------
     setlogid
     set the log id, 
     if the slide and main log files are the same
        # this is batch process not a slide process
     ------------------------------------------
     Usage: $setlogid($cmodule)
    ----------------------------------------- #>
    [string]setlogid($cmodule){
        #
        if ($this.moduleinfo.($cmodule).slidelog -match `
            [regex]::Escape($this.moduleinfo.($cmodule).mainlog)){
            $ID= $this.BatchID
        } else {
            $ID = $this.slideid
        }
        return $ID
        #
    }
    <# -----------------------------------------
     setlogvers
     set the log version number to match
     ------------------------------------------
     Usage: $setlogvers($cmodule)
    ----------------------------------------- #>
    [string]setlogvers($cmodule){
        #
        $vers = $this.moduleinfo.($cmodule).vers -replace 'v', ''
        $vers = ($vers -split '\.')[0,1,2] -join '.'
        return $vers
        #
    }
    <# -----------------------------------------
     deflogstatus
     if there was an error return true 
     if not a dependency check and the latest 
        run is finished return true
     if it is a dependency check and 
        the lastest run is not finished return true
     ------------------------------------------
     Usage: $deflogstatus($startdate, $finishdate, $errorline, $dependency)
    ----------------------------------------- #>
    [array]deflogstatus($startdate, $finishdate, $errorline, $dependency){
        #
        $errorlogical = $this.errorlogical($startdate, $finishdate, $errorline)
        #
        if ( (!$startdate) -or ($startdate -eq $this.empty_time) -or
            $errorlogical -or (
                !$dependency -and ($finishdate -ne $this.empty_time) -and
                (get-date $finishdate) -gt (get-date $startdate)
            ) -or (
                $dependency -and (($finishdate -eq $this.empty_time) -or
                (get-date $finishdate) -le (get-date $startdate))
            )
        ){
            if ($errorlogical){
                return @($true, $errorline)
            } else {
                return @($true)
            }
        } else {
            return @($false)
        }
    }
    #
    [switch]errorlogical($startdate, $finishdate, $errorline){
        #
        $errordate = $errorline.Date
        $errorlogical = (
            $startdate -and $startdate -ne $this.empty_time -and
            $errordate -and $errordate -ne $this.empty_time -and
            $finishdate -and $finishdate -ne $this.empty_time -and
            (get-date $startdate) -le (get-date $errordate) -and 
            (get-date $finishdate) -ge (get-date $errordate)
        )
        return $errorlogical
        #
    }
    <# -----------------------------------------
     checkscan
     check that the slide has been scanned,
     all scan products exist and the 
     slides are ready to be transferred.
    ------------------------------------------
     Input: 
        - log[mylogger]: astropath log object
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module needs to be run,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkscan(log, dependency)
    ----------------------------------------- #>
    [int]checkscan(){
        #
        if (!($this.moduleinfo.transfer.version -match '0.0.1') -and 
            $this.checklog('transfer', $true)){
            return 2
        }
        #
        if (!$this.testtransferfiles()){
            return 2
        }
        #
        $im3s = (Get-ChildItem ($this.Scanfolder() + '\MSI\*') *im3).Count
        #    
        if (!$im3s){
            return 2
        }
        #
        return 3
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
     returns 2 if current module needs to be run,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checktransfer(log, dependency)
    ----------------------------------------- #>
    [int]checktransfer(){
        #
        if (!($this.moduleinfo.transfer.version -match '0.0.1') -and 
            $this.checklog('transfer', $true)){
            return 2
        }
        #
        if (!$this.testtransferfiles()){
            return 2
        }
        #
        $im3s = (Get-ChildItem ($this.Scanfolder() + '\MSI\*') *im3).Count
        #    
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
    [int]checkshredxml(){
        #
        $cmodule = 'shredxml'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }  
        #
        if (!$this.('test' + $cmodule + 'files')()){
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
    [int]checkmeanimage(){
        #
        $cmodule = 'meanimage'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }  
        #
        if (!$this.('test' + $cmodule + 'files')()){
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
    [int]checkbatchmicomp(){
        #
        $cmodule = 'batchmicomp'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.moduleinfo.($cmodule).vers -match '0.0.1'){
            return 3
        }
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }  
        #
        if (!$this.('test' + $cmodule + 'files')()){
            return 2
        }
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
    [int]checkbatchflatfield(){
        #
        $cmodule = 'batchflatfield'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        }
        #
        if ($this.moduleinfo.batchflatfield.vers -notmatch '0.0.1'){
            <#
            #
            if ($this.moduleinfo.batchmicomp.status -ne 'FINISHED'){
                return 1
            }
            #
            if ($this.moduleinfo.meanimage.status -ne $this.status.finish){
                return 1
            }
            #>
            if ($this.teststatus){
                $ids = $this.ImportCorrectionModels($this.mpath, $false)
            } else{ 
                $ids = $this.ImportCorrectionModels($this.mpath)
            }
            #
            if ($ids.slideid -notcontains $this.slideid){
                return 2
            }
            #
            if (!$this.testpybatchflatfield()){
                return 2
            }
            #
        } else {
            <#
            if ($this.moduleinfo.meanimage.status -ne $this.status.finish){
                return 1
            }
            #>
            if ($this.checklog($cmodule, $true)){
                return 2
            }
            #
            #
            if (!$this.testbatchflatfield()){
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
    [int]checkwarpoctets(){
        #
        $cmodule = 'warpoctets'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.moduleinfo.($cmodule).vers -match '0.0.1'){
            return 3
        }
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }  
        #
        if (!$this.('test' + $cmodule + 'files')()){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkbatchwarpkeys
     check that the batch warp keys module has completed
     and all products exist for the batch
    ------------------------------------------
     Input: 
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module needs to be run,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkbatchwarpkeys(dependency)
    ----------------------------------------- #>
    [int]checkbatchwarpkeys(){
        #
        $cmodule = 'batchwarpkeys'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.moduleinfo.($cmodule).vers -match '0.0.1'){
            return 3
        }
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }  
        #
        if (!$this.('test' + $cmodule + 'files')()){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkbatchwarpfits
     check that the batch warp fits module has completed
     and all products exist for the batch
    ------------------------------------------
     Input: 
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module needs to be run,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkbatchwarpfits(dependency)
    ----------------------------------------- #>
    [int]checkbatchwarpfits(){
        #
        $cmodule = 'batchwarpfits'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.moduleinfo.($cmodule).vers -match '0.0.1'){
            return 3
        }
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }  
        #
        if (!$this.('test' + $cmodule + 'files')()){
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
    [int]checkimagecorrection(){
        #
        $cmodule = 'imagecorrection'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }  
        #
        if (!$this.('test' + $cmodule + 'files')()){
            return 2
        }
        #
        return 3
    }
    <# -----------------------------------------
     checkvminform
     checks the status of a particular antibody
     and if it is ready to be run. Details on
     checks follows:
     check the image correction step has 
     finished. check that the slide - antibody
     pair is in the local queues (add if not)
     check that the antibody is in the queue
     w/o an algorithm. checks the log for
     errors. checks for tasks with algorithm
     and ready to be run.
    ------------------------------------------
     Input: 
        - antibody[string]: antibody to check
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkvminform(dependency)
    ----------------------------------------- #>
    [int]checkvminform($antibody){
        #
        $cmodule = 'vminform'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.vmq.checkfornewtask($this.project, 
            $this.slideid, $antibody)){
                return 1
        }
        #
        if ($this.vmq.checkforidletask($this.project, 
            $this.slideid, $antibody)){
            return 1
        }
        #
        $taskid = $this.vmq.checkforreadytask($this.project, 
            $this.slideid, $antibody)
        #
        if ($taskid){
            return 2
        }
        #
        $logoutput = $this.checklog($cmodule, $antibody, $true)
        if ($logoutput[1]){
            return 5
        } elseif ($logoutput){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkmerge
     checks if the merge function needs to be 
     run. Details on checks follows:
     checks all vminform ab logs for finished,
     checks the mergelog for previously run,
     tests the mergefiles exist and the dates
     are newer than the most recent inform run.
    ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete.
     ------------------------------------------
     Usage: $this.checkmerge(dependency)
    ----------------------------------------- #>
    [int]checkmerge(){
        #
        $cmodule = 'merge'
        #
        $this.getantibodies()
        #
        foreach ($abx in $this.antibodies){
            #
            if ($this.moduleinfo.vminform.($abx).status -ne $this.status.finish){
                return 1
            }
            #
        }
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }
        #
        if (!$this.testmergefiles($this.antibodies)){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkimageqa
     check if image qa has finished or not. 
     Details on checks follows:
     first check that the slide has finished 
     previous steps then check that the slides
     have been checked off manually in the 
     image qa spreadsheet. 
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkimageqa(dependency)
    ----------------------------------------- #>
    [int]checkimageqa(){
        #
        $cmodule = 'imageqa'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        $this.getantibodies()
        #
        if ($this.checknewimageqa($this.antibodies)){
            return 2
        }
        #
        if(!$this.testimageqafile($this.antibodies)){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checksegmaps
     check if the segmaps step should be run. 
     details on the checks follow:
     first check that the previous steps have
     finished then check for issues in the log,
     finially check that there are as many seg
     files as im3 files.
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checksegmaps(dependency)
    ----------------------------------------- #>
    [int]checksegmaps(){
        #
        $cmodule = 'segmaps'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }
        #
        if (!$this.('test' + $cmodule + 'files')()){
            return 2
        }
        #
        return 3
        #
    }
     <# -----------------------------------------
     checkdbload
     check if the slide is ready to be loaded 
     into the database by checking all previous
     steps.
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checksegmaps(dependency)
    ----------------------------------------- #>
    [int]checkdbload(){
        #
        $cmodule = 'dbload'
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        if ($this.checklog($cmodule, $true)){
            return 2
        }
        #
        return 3
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
        $batchescomplete = @()
        #
        $batcharrayunique | foreach-object {
            $nslidescomplete = ($batcharray -match $_).count
            $projectbatchpair = $_ -split ','
            $sample = sampledef -mpath $this.mpath -module $cmodule `
                -batchid $projectbatchpair[1] -project $projectbatchpair[0]
            $nslidesbatch = $sample.batchslides.count
            if ($nslidescomplete -eq $nslidesbatch){
                $batchescomplete += $_
            }
        }
        return $batchescomplete
    }
    #
    [string]previousstep($step){
        #
        $id = $($this.pipeline_steps.Values).indexOf($step)
        if ($id -eq 0){
            return $step
        } else {
            $newid = $id - 1
            return $this.pipeline_steps[$newid]
        }
        #
    }
    #
    [switch]checkpreviousstep($step){
        #
        $previousstep = $this.previousstep($step)
        #
        if ($this.moduleinfo.($previousstep).status -ne $this.status.finish){
            return $true
        } 
        #
        return $false
        #
    }
}