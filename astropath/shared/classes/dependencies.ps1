<# -------------------------------------------
 dependencies
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build the sample trackers for each
 sample and module
 -------------------------------------------#>
 class dependencies : sampledef {
    #
    [hashtable]$modulestatus = @{}
    #
    dependencies($mpath, $slideid): base ($mpath, '', $slideid){}
    #
    # dependencies($mpath, $module, $batchid, $project) : base ($mpath, $module, $batchid, $project){}
    #
    [void]getlogstatus($cmodule){
        #
        $logoutput = $this.checklog($cmodule, $false)
        #
        if ($logoutput[1]){
            $this.modulestatus.($cmodule) = $logoutput[1].Message
        } elseif ($logoutput) {
            #
            $statusval = ($this.('check'+$cmodule)($false))
            if ($statusval -eq 1){
                $this.modulestatus.($cmodule) = 'WAITING'
            } elseif ($statusval -eq 2){
                $this.modulestatus.($cmodule) = 'READY'
            } elseif ($statusval -eq 3){
                $this.modulestatus.($cmodule) = 'FINISHED'
            } elseif ($statusval -eq 4) {
                $this.modulestatus.($cmodule) = 'NA'
            } else {
                $this.modulestatus.($cmodule) = 'UNKNOWN'
            }
            #
        } else {
            #
            $this.modulestatus.($cmodule) = 'RUNNING'
            #
        }
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
    [array]checklog($cmodule, $dependency){
        #
        if (!(test-path $this.modulelogs.($cmodule).slidelog)){
            return @($true)
        }
        #
        $loglines = $this.importlogfile($this.modulelogs.($cmodule).slidelog)
        $vers = $this.setlogvers($cmodule)
        $ID = $this.setlogid($cmodule)
        #
        $startdate = ($this.selectlogline($loglines, $ID, 'START', $vers)).Date
        $finishdate = ($this.selectlogline($loglines, $ID, 'FINISH', $vers)).Date
        $errorline = $this.selectlogline($loglines, $ID, 'ERROR', $vers)
        #
        return ($this.deflogstatus($startdate, $finishdate, $errorline, $dependency))
        #
    }
    #
    [array]checklog($cmodule, $antibody, $dependency){
        #
        if (!(test-path $this.modulelogs.($cmodule).slidelog)){
            return @($true)
        }
        #
        $loglines = $this.importlogfile($this.modulelogs.($cmodule).slidelog)
        $vers = $this.setlogvers($cmodule)
        $ID = $this.setlogid($cmodule)
        #
        $startdate = ($this.selectlogline($loglines, $ID, 'START', $vers, $antibody)).Date
        $finishdate = ($this.selectlogline($loglines, $ID, 'FINISH', $vers, $antibody)).Date
        $errorline = $this.selectlogline($loglines, $ID, 'ERROR', $vers, $antibody)
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
        if ($this.modulelogs.($cmodule).slidelog -match `
            [regex]::Escape($this.modulelogs.($cmodule).mainlog)){
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
        $vers = $this.modulelogs.($cmodule).vers -replace 'v', ''
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
        $errordate = $errorline.Date
        $errorlogical = ($startdate -le $errordate -and $finishdate -ge $errordate) 
        #
        if ( !$startdate -or $errorlogical -or 
            (!$dependency -and ($finishdate -gt $startdate)) -or 
            ($dependency -and !($finishdate -gt $startdate))
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
    [int]checktransfer($dependency){
        #
        if (!($this.modulelogs.transfer.vers -match '0.0.1') -and 
            $this.checklog('transfer', $true)){
            return 2
        }
        #
        # check for checksum, qptiff, and annotationxml
        #
        $file = $this.CheckSumsfile()
        $file2 = $this.qptifffile()
        $file3 = $this.annotationxml()
        $im3s = (gci ($this.Scanfolder() + '\MSI\*') *im3).Count
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
    [int]checkshredxml($dependency){
        #
        if (!($this.checktransfer($true) -eq 3)){
            return 1
        }
        #
        if ($this.checklog('shredxml', $true)){
            return 2
        }
        #
        if (!$this.testxmlfiles()){
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
    [int]checkmeanimage($dependency){
        #
        if (!($this.checkshredxml($true) -eq 3)){
            return 1
        }
        #
        if ($this.checklog('meanimage', $true)){
            return 2
        }
        #
        if (!$this.testmeanimagefiles()){
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
    [int]checkbatchmicomp($dependency){
        #
        # if task is not a dependency and the version is
        # 0.0.1 then just checkout
        #
        if (
            !$dependency -and
             $this.modulelogs.batchmicomp.vers -match '0.0.1'
            ){
            return 4
        }
        #
        if (!($this.checkmeanimage($true) -eq 3)){
            return 1
        }
        #
        if ($this.checklog('batchmicomp', $true)){
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
    [int]checkbatchflatfield($dependency){
        #
        # if task is not a dependency and the version is
        # not 0.0.1 then just checkout
        #
        if (
            !$dependency -and
             $this.modulelogs.batchflatfield.vers -notmatch '0.0.1'
            ){
            return 4
        }
        #
        # if the version is not 0.0.1 in batchflatfield, do meanimagecomparison
        # instead
        if ($this.modulelogs.batchflatfield.vers -notmatch '0.0.1'){
            #
            if (!($this.checkbatchmicomp($true) -eq 3)){
                return 1
            }
            #
            $ids = $this.ImportCorrectionModels($this.mpath)
            if ($ids.slideid -notcontains $this.slideid){
                return 2
            }
            #
            if (!$this.testpybatchflatfield()){
                return 2
            }
            #
        } else {
            #
            if (!($this.checkmeanimage($true) -eq 3)){
                return 1
            }
            #
            if ($this.checklog('batchflatfield', $true)){
                return 2
            }
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
    [int]checkwarpoctets($dependency){
        #
        if (!($this.checkbatchflatfield($true) -eq 3)){
            return 1
        }
        #
        if ($this.checklog('warpoctets', $true)){
            return 2
        }
        #
        if (!$this.testwarpoctets()){
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
    [int]checkimagecorrection($dependency){
        #
        if (!($this.checkbatchflatfield($true) -eq 3)){
            return 1
        }
        #
        if ($this.checklog('imagecorrection', $true)){
            return 2
        }
        #
        if(!$this.testimagecorrectionfiles()){
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
     checkvminform
     place holder
    ------------------------------------------
     Input: 
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkvminform(dependency)
    ----------------------------------------- #>
    [int]checkvminform($dependency){
        #
        return 4
        #
    }
    <# -----------------------------------------
     checkmergeloop
     place holder
    ------------------------------------------
     Input: 
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkmergeloop(dependency)
    ----------------------------------------- #>
    [int]checkmergeloop($dependency){
        #
        return 4
        #
    }
    <# -----------------------------------------
     checksegmaps
     place holder
    ------------------------------------------
     Input: 
        - dependency[switch]: true or false
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checksegmaps(dependency)
    ----------------------------------------- #>
    [int]checksegmaps($dependency){
        #
        return 4
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