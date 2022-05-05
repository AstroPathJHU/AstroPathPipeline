<# -------------------------------------------
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
    [string]$specialcasedeps = '^' + (@('scan','scanvalidation','transfer',
        'batchmicomp','merge','imageqa') -join '$|^') + '$'
    #
    [string]$empty_time = 'NA'
    #
    dependencies($mpath): base ($mpath){$this.initdependencies()}
    #
    dependencies($mpath, $slideid): base ($mpath, '', $slideid){$this.initdependencies()}
    #
    [void]initdependencies(){
        $this.importdependencyinfo($this.mpath) | Out-Null
    }
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
            if ($cmodule -match $this.specialcasedeps){
                $check = $this.('check'+$cmodule)()
            } else {
                $check = $this.checkmodule($cmodule)
            }
            #
            switch ($check){
                1 {
                    $this.moduleinfo.($cmodule).status = $this.status.waiting
                    $this.moduleinfo.($cmodule).starttime = $this.empty_time
                    $this.moduleinfo.($cmodule).finishtime = $this.empty_time
                }
                2 {
                    $this.moduleinfo.($cmodule).status = $this.status.ready
                    $this.moduleinfo.($cmodule).starttime = $this.empty_time
                    $this.moduleinfo.($cmodule).finishtime = $this.empty_time
                }
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
        #
        if ($logoutput){
            switch ($this.('check'+$cmodule)($antibody)){
                1 {
                    $this.moduleinfo.($cmodule).($antibody).status = $this.status.waiting
                    $this.moduleinfo.($cmodule).($antibody).starttime = $this.empty_time
                    $this.moduleinfo.($cmodule).($antibody).finishtime = $this.empty_time
                }
                2 {
                    $this.moduleinfo.($cmodule).($antibody).status = $this.status.ready
                    $this.moduleinfo.($cmodule).($antibody).starttime = $this.empty_time
                    $this.moduleinfo.($cmodule).($antibody).finishtime = $this.empty_time
                }
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
        return $this.checklog($cmodule, '', $dependency)
        #
    }
    #
    [array]checkloginit($cmodule, $dependency){
        #
        return ($this.checkloginit($cmodule, '', $dependency))
        #
    }
    #
    [array]checklog($cmodule, $antibody, $dependency){
        #
        if (!($this.modulelogs.($cmodule).($this.project))){
            return @($true)
        }
        #
        if ($antibody){
            $cmoduleinfo =  $this.moduleinfo.($cmodule).($antibody)
        } else {
            $cmoduleinfo =  $this.moduleinfo.($cmodule)
        }
        #
        $startdate = $cmoduleinfo.StartTime
        $finishdate = $cmoduleinfo.FinishTime
        $errorline = $cmoduleinfo.Errorline
        #
        return ($this.deflogstatus($startdate,
            $finishdate, $errorline, $dependency))
        #
    }
    #
    [array]checkloginit($cmodule, $antibody, $dependency){
        #
        if ($antibody){
            $cmoduleinfo =  $this.moduleinfo.($cmodule).($antibody)
        } else {
            $cmoduleinfo =  $this.moduleinfo.($cmodule)
        }
        #
        if (!($this.modulelogs.($cmodule).($this.project))){
            $cmoduleinfo.StartTime = $this.empty_time
            $cmoduleinfo.FinishTime = $this.empty_time
            return @($true)
        }
        #
        $loglines = $this.modulelogs.($cmodule).($this.project)
        $vers = $this.setlogvers($cmodule)
        $ID = $this.setlogid($cmodule)
        #
        if ($antibody){
            $filteredloglines = $this.filterloglines($loglines, $ID, $vers, $antibody)
        } else {
            $filteredloglines = $this.filterloglines($loglines, $ID, $vers)
        }
        #
        if ($filteredloglines.startdate){
            $cmoduleinfo.StartTime = $filteredloglines.startdate
        } else {
            $cmoduleinfo.StartTime = $this.empty_time
        }
        #
        if ($filteredloglines.startdate -and 
            $filteredloglines.finishdate -and
            (get-date $filteredloglines.finishdate) -gt 
                (get-date $filteredloglines.startdate)
        ){
                $cmoduleinfo.FinishTime = $filteredloglines.finishdate
        } else {
            $cmoduleinfo.FinishTime = $this.empty_time
        }
        #
        $cmoduleinfo.Errorline = $filteredloglines.errorline
        #
        return ($this.deflogstatus($filteredloglines.startdate,
            $filteredloglines.finishdate, $filteredloglines.errorline, $dependency))
        #
    }
    #
    [hashtable]filterloglines($loglines, $ID, $vers){
        $startdate = ($this.selectlogline($loglines,
            $ID, $this.log_start, $vers)).Date[0..18] -join ''
        if ($startdate){
            $startdate = $startdate.Date[0..18] -join ''
        }
        $finishdate = ($this.selectlogline($loglines,
            $ID, $this.log_finish, $vers)).Date[0..18] -join ''
        if ($finishdate){
            $finishdate = $finishdate.Date[0..18] -join ''
        }
        $errorline = $this.selectlogline($loglines,
            $ID, $this.log_error)
        $filteredloglines = @{
            startdate = $startdate;
            finishdate = $finishdate;
            errorline = $errorline
        }
        return $filteredloglines
    }
    #
    [hashtable]filterloglines($loglines, $ID, $vers, $antibody){
        $startdate = ($this.selectlogline($loglines,
            $ID, $this.log_start, $vers, $antibody)).Date[0..18] -join ''
        if ($startdate){
            $startdate = $startdate.Date[0..18] -join ''
        }
        $finishdate = ($this.selectlogline($loglines,
            $ID, $this.log_finish, $vers, $antibody)).Date[0..18] -join ''
        if ($finishdate){
            $finishdate = $finishdate.Date[0..18] -join ''
        }
        $errorline = $this.selectlogline($loglines,
            $ID, $this.log_error, '', $antibody)
        $filteredloglines = @{
            startdate = $startdate;
            finishdate = $finishdate;
            errorline = $errorline
        }
        return $filteredloglines
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
                !$dependency -and $finishdate -and 
                ($finishdate -ne $this.empty_time) -and
                (get-date $finishdate) -gt (get-date $startdate)
            ) -or (
                $dependency -and (!$finishdate -or 
                ($finishdate -eq $this.empty_time) -or
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
        if ($errordate){
            $errordate = $errorline.Date[0..18] -join ''
        }
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
    #
    [int]checkscanvalidation(){
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
        $check = $this.versiondependentchecks($cmodule)
        #
        if ($check){
            return $check
        } 
        <#
        if ($this.checklog($cmodule, $true)){
            return 2
        }  
        #
        if (!$this.('test' + $cmodule + 'files')()){
            return 2
        }
        #>
        return 3
        #
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
        $this.moduleinfo.($cmodule).($antibody).taskid = $taskid
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
    #
    [int]checkmodule($cmodule){
        #
        if ($this.checkpreviousstep($cmodule)){
            return 1
        } 
        #
        $check = $this.versiondependentchecks($cmodule)
        #
        if ($check){
            return $check
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
    #
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
    [switch]checkpreviousstep($step){
        #
        $previousstep = ($this.dependency_data |
            Where-Object {$_.module -contains $step}).dependency
        #
        if ($this.moduleinfo.($previousstep).status -ne $this.status.finish){
            return $true
        } 
        #
        return $false
        #
    }
    #
    [int]versiondependentchecks($cmodule){
        #
        $vers = $this.moduleinfo.($cmodule).vers
        if ($vers -match '0.0.1'){
            switch -regex ($cmodule){
                batchmicomp {return 3}
                batchflatfield{
                    if ($this.checklog($cmodule, $true)){
                        return 2
                    }
                }
                warpoctets {return 3}
                batchwarpkeys {return 3}
                batchwarpfits {return 3}

            }
        }
        #
        return 0
        #
    }
    #
}