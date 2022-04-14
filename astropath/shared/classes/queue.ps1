﻿<# -------------------------------------------
 queue
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build the queues and check 
 dependencies
 -------------------------------------------#>
class queue : vminformqueue{
    #
    [Array]$originaltasks
    [Array]$cleanedtasks
    [string]$queue_file
    [string]$informvers
    [string]$project
    #
    queue($module){
        $this.init('\\bki04\astropath_processing', $module,'', '')
    }
    queue($mpath, $module){
        $this.init($mpath, $module, '', '')
    }
    queue($mpath, $module, $project){
        $this.init($mpath, $module, $project, '')
    }
    queue($mpath, $module, $project, $slideid){
        $this.init($mpath, $module, $project, $slideid)
    }
    #
    init($mpath, $module, $project, $slideid){
        #
        $this.mpath = $mpath
        $this.module = $module 
        $this.project = $project
        $this.slideid = $slideid.trim()
        $this.importaptables($this.mpath, $true)
        #
    }
    <# -----------------------------------------
     ExtractQueue
     get the queue either from the file (vminform)
     or from the dependency checks
     ------------------------------------------
     Usage: $this.ExtractQueue()
    ----------------------------------------- #>
    [void]ExtractQueue(){
        #
        if ('vminform' -ne $this.module){
            $this.buildqueue()
        } else {
            $this.('check'+$this.module)()
            #$this.coalescevminformqueues()
        }
        #
    }
    <# -----------------------------------------
     buildqueue
     build the queue from the dependency checks
     ------------------------------------------
     Usage: $this.buildqueue()
    ----------------------------------------- #>
    [void]buildqueue(){
        #
        $slides = $this.importslideids($this.mpath)
        #
        # select samples from the appropriate modules 
        #
        $projects = $this.getapprojects($this.module)
        #
        $cleanedslides = $slides | 
            Where-Object {$projects -contains $_.Project}
        #
        $slidesnotcomplete = $this.defNotCompletedSlides($cleanedslides)
        $slidearray = New-Object 'object[]' $slidesnotcomplete.count
        $batcharray = New-Object 'object[]' $slidesnotcomplete.count
        #
        if ($slidesnotcomplete.count -eq 1){
            $slidearray[0] = ($slidesnotcomplete.Project,
                $slidesnotcomplete.Slideid)
            $batcharray[0] = ($slidesnotcomplete.Project, 
                $slidesnotcomplete.Slideid)
        } else {
            for($i=0; $i -lt $slidesnotcomplete.count;$i++){
                $slidearray[$i] = ($slidesnotcomplete.Project[$i],
                    $slidesnotcomplete.Slideid[$i])
                $batcharray[$i] = ($slidesnotcomplete.Project[$i],
                    $slidesnotcomplete.BatchID[$i])
            }
        }
        #
        if ($this.module -match 'batch'){
            $slidearray = $this.AggregateBatches($batcharray)
        }
        #
        $this.cleanedtasks = $slidearray
        #
    }
    <# -----------------------------------------
     defNotCompletedSlides
     For each slide, check the current module 
     and the module dependencies to see if the
     slide needs to be run through the module
     ------------------------------------------
     Usage: $this.defNotCompletedSlides(cleanedslides)
    ----------------------------------------- #>
    [array]defNotCompletedSlides($cleanedslides){
        #
        $slidesnotcomplete = @()
        $c = 0
        $ctotal = $cleanedslides.count
        if (!($cleanedslides)){
            return $slidesnotcomplete
        }
        #
        $log = [mylogger]::new($this.mpath, $this.module, $cleanedslides[0].slideid)
        #
        foreach($slide in $cleanedslides){
            #
            $p = [math]::Round(100 * ($c / $ctotal))
            Write-Progress -Activity "Checking slides" `
                           -Status ("$p% Complete : Checking " + $slide.slideid) `
                           -PercentComplete $p 
            $c += 1 
            #
            $log.Sample($slide.slideid, $this.module, $cleanedslides)
            $log.vers = $log.GetVersion($this.mpath, $this.module, $log.project, 'short')
            #
            if ($this.module -match 'batch'){
                $log.slidelog = $log.mainlog
            }
            #
            if ($this.checklog($log, $false)){
                #
                if (($this.('check'+$this.module)($log, $false) -eq 2)) {
                    $slidesnotcomplete += $slide
                }
                #
            }
        }
        #
        write-progress -Activity "Checking slides" -Status "100% Complete:" -Completed
        #
        return $slidesnotcomplete
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
        $loglines = $this.opencsvfile($log.slidelog, ';',
            @('Project','Cohort','slideid','Message','Date'))
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
                         ($_.Slideid -contains $ID) -and 
                         ($_.Message -match $statustype)
                    } |
                    Select-Object -Last 1 
        }
        #
        $d1 = ($savelog | Where-Object {$_.Message -match $statustypes[0]}).Date
        $d2 = ($loglines |
                 Where-Object {
                    $_.Message -match $statustypes[1] -and
                     ($_.Slideid -contains $ID)
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
    [mylogger]updatelogger([mylogger]$log, $module){
        #
        $log.module = $module
        $log.deflogpaths()
        $log.vers = $log.GetVersion($this.mpath, $module, $log.project, 'short')
        return $log
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
        $log = $this.updatelogger($log, 'transfer')
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
        $im3s = (get-childitem ($log.Scanfolder() + '\MSI\*') *im3).Count
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
        $log = $this.updatelogger($log, 'shredxml')
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
        $log = $this.updatelogger($log, 'meanimage')
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
        $log = $this.updatelogger($log, 'batchmicomp')
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
        $log = $this.updatelogger($log, 'batchflatfield')
        #
        # if the version is not 0.0.1 in batchflatfield, do meanimagecomparison
        # instead
        if ($log.vers -notmatch '0.0.1'){
            <#
            if (!($this.checkbatchmicomp($log, $true) -eq 3)){
                return 1
            }
            #>
            if (!($this.checkmeanimage($log, $true) -eq 3)){
                return 1
            }
            #
            $log = $this.updatelogger($log, 'batchflatfield')
            #
            $ids = $this.ImportCorrectionModels($this.mpath)
            if ($ids.slideid -notcontains $log.slideid){
                return 1
            }
            #
            if (!$log.testpybatchflatfield()){
                return 1
            }
            #
        } else {
            #
            if (!($this.checkmeanimage($log, $true) -eq 3)){
                return 1
            }
            #
            $log = $this.updatelogger($log, 'batchflatfield')
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
        $log = $this.updatelogger($log, 'warpoctets')
        #
        if ($log.vers -match '0.0.1'){
            RETURN 3
        }
        #
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
     checkbatchwarpkeys
     check that the batch warp keys module has completed
     and all products exist for the batch
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
    [int]checkbatchwarpkeys([mylogger]$log, $dependency){
        #
        if (!($this.checkwarpoctets($log, $true) -eq 3)){
            return 1
        }
        #
        $log = $this.updatelogger($log, 'batchwarpkeys')
        #
        if ($log.vers -match '0.0.1'){
            RETURN 3
        }
        #
        if ($this.checklog($log, $true)){
            return 2
        }
        #
        if (!$log.testbatchwarpkeysfiles()){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
    checkbatchwarpfits
    check that the batch warp keys module has completed
    and all products exist for the batch
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
    [int]checkbatchwarpfits([mylogger]$log, $dependency){
        #
        if (!($this.checkbatchwarpkeys($log, $true) -eq 3)){
            return 1
        }
        #
        $log = $this.updatelogger($log, 'batchwarpfits')
        #
        if ($log.vers -match '0.0.1'){
            RETURN 3
        }
        #
        if ($this.checklog($log, $true)){
            return 2
        }
        #
        if (!$log.testbatchwarpfitsfiles()){
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
        $log = $this.updatelogger($log, 'imagecorrection')
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
        $this.informvers = '2.4.8'
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
        $tempcleanedtasks = $this.originaltasks -replace ('\s','')
        #
        $this.cleanedtasks = New-Object 'object[]' $tempcleanedtasks.count
        $cnt = 0
        #
        foreach ($newtask in $tempcleanedtasks){
           $this.cleanedtasks[$cnt] = ($newtask.split(',')[0..3])
           $cnt ++
        }
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
    [array]Aggregatebatches($batcharray){
        $batcharrayunique = $batcharray | Sort-Object | Get-Unique
        $batchescomplete = New-Object 'object[]' $batcharrayunique.count
        $cnt = 0
        #
        $batcharrayunique | foreach-object {
            $nslidescomplete = ($batcharray -match $_).count
            $sample = sample -mpath $this.mpath -module $this.module -batchid $_[1] -project $_[0]
            $nslidesbatch = $sample.batchslides.count
            if ($nslidescomplete -eq $nslidesbatch){
                $batchescomplete[$cnt] = $_
                $cnt ++
            }
        }
        #
        return ($batchescomplete -ne $null)
    }
    #
    [void]handleAPevent($file){
        #
        $fpath = Split-Path $file
        $file = Split-Path $file -Leaf
        #
        switch -regex ($file){
            $this.cohorts_file {$this.importcohortsinfo($this.mpath, $false)}
            $this.paths_file {$this.importcohortsinfo($this.mpath, $false)}
            $this.config_file {$this.ImportConfigInfo($this.mpath, $false)}
            $this.slide_file {$this.ImportSlideIDs($this.mpath, $false)}
            $this.ffmodels_file {$this.ImportFlatfieldModels($this.mpath, $false)}
            $this.corrmodels_file {$this.ImportCorrectionModels($this.mpath, $false)}
            $this.micomp_file {$this.ImportMICOMP($this.mpath, $false)}
            $this.worker_file {$this.Importworkerlist($this.mpath, $false)}
            #$this.vmq.mainqueue_file {$this.vmq.openmainqueue($false)}
            #$this.vmq.localqueue_file {}
        }
    }
}