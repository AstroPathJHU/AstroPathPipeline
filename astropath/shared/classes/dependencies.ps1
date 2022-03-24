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
    dependencies($mpath): base ($mpath){}
    #
    dependencies($mpath, $slideid): base ($mpath, '', $slideid){}
    #
    # dependencies($mpath, $module, $batchid, $project) : base ($mpath, $module, $batchid, $project){}
    #
    [void]getlogstatus($cmodule){
        if ($cmodule -match 'vminform'){
            #
            $this.getantibodies()
            $this.antibodies | ForEach-Object{
                $this.getlogstatussub($cmodule, $_)
            }
            #
        } else {
            $this.getlogstatussub($cmodule)
        }
    }
    #
    [void]getantibodies(){
       # try{
            $this.findantibodies($this.basepath)
       # } catch {
        #    Write-Host $_.Exception.Message
        #    return
        #}
    }
    #
    [void]getlogstatussub($cmodule){
        #
        $logoutput = $this.checklog($cmodule, $false)
        #
        if ($logoutput[1]){
            $this.moduleinfo.($cmodule).status = $logoutput[1].Message
        } elseif ($logoutput) {
            #
            $statusval = ($this.('check'+$cmodule)())
            if ($statusval -eq 1){
                $this.moduleinfo.($cmodule).status = 'WAITING'
            } elseif ($statusval -eq 2){
                $this.moduleinfo.($cmodule).status = 'READY'
            } elseif ($statusval -eq 3){
                $this.moduleinfo.($cmodule).status = 'FINISHED'
            } elseif ($statusval -eq 4) {
                $this.moduleinfo.($cmodule).status = 'NA'
            } else {
                $this.moduleinfo.($cmodule).status = 'UNKNOWN'
            }
            #
        } else {
            #
            $this.moduleinfo.($cmodule).status = 'RUNNING'
            #
        }
        #
    }
    #
    [void]getlogstatussub($cmodule, $antibody){
        #
        $logoutput = $this.checklog($cmodule, $antibody, $false)
        $this.moduleinfo.($cmodule).($antibody) = @{}
        #
        if ($logoutput[1]){
            $this.moduleinfo.($cmodule).($antibody).status = $logoutput[1].Message
        } elseif ($logoutput) {
            #
            $statusval = ($this.('check'+$cmodule)($antibody))
            if ($statusval -eq 1){
                $this.moduleinfo.($cmodule).($antibody).status = 'WAITING'
            } elseif ($statusval -eq 2){
                $this.moduleinfo.($cmodule).($antibody).status = 'READY'
            } elseif ($statusval -eq 3){
                $this.moduleinfo.($cmodule).($antibody).status = 'FINISHED'
            } elseif ($statusval -eq 4) {
                $this.moduleinfo.($cmodule).($antibody).status = 'NA'
            } else {
                $this.moduleinfo.($cmodule).($antibody).status = 'UNKNOWN'
            }
            #
        } else {
            #
            $this.moduleinfo.($cmodule).($antibody).status = 'RUNNING'
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
        if (!(test-path $this.moduleinfo.($cmodule).slidelog)){
            return @($true)
        }
        #
        $loglines = $this.importlogfile($this.moduleinfo.($cmodule).slidelog)
        $vers = $this.setlogvers($cmodule)
        $ID = $this.setlogid($cmodule)
        #
        $startdate = ($this.selectlogline($loglines, $ID, 'START', $vers)).Date
        $finishdate = ($this.selectlogline($loglines, $ID, 'FINISH', $vers)).Date
        $errorline = $this.selectlogline($loglines, $ID, 'ERROR')
        #
        return ($this.deflogstatus($startdate, $finishdate, $errorline, $dependency))
        #
    }
    #
    [array]checklog($cmodule, $antibody, $dependency){
        #
        if (!(test-path $this.moduleinfo.($cmodule).slidelog)){
            return @($true)
        }
        #
        $loglines = $this.importlogfile($this.moduleinfo.($cmodule).slidelog)
        $vers = $this.setlogvers($cmodule)
        $ID = $this.setlogid($cmodule)
        #
        $startdate = ($this.selectlogline($loglines, $ID, 'START', $vers, $antibody)).Date
        $finishdate = ($this.selectlogline($loglines, $ID, 'FINISH', $vers, $antibody)).Date
        $errorline = $this.selectlogline($loglines, $ID, 'ERROR', '', $antibody)
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
        if ($this.moduleinfo.transfer.status -ne 'FINISHED'){
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
    [int]checkmeanimage(){
        #
        if ($this.moduleinfo.shredxml.status -ne 'FINISHED'){
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
    [int]checkbatchmicomp(){
        #
        # if task is not a dependency and the version is
        # 0.0.1 then just checkout
        #
        if ($this.moduleinfo.meanimage.status -ne 'FINISHED'){
            return 1
        }
        #
        if (
             $this.moduleinfo.batchmicomp.vers -match '0.0.1'
            ){
            return 3
        }
        #
        if ($this.checklog('batchmicomp', $true)){
            return 2
        }
        #
        if (!$this.testbatchmicompfiles()){
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
        if ($this.moduleinfo.batchflatfield.vers -notmatch '0.0.1'){
            <#
            #
            if ($this.moduleinfo.batchmicomp.status -ne 'FINISHED'){
                return 1
            }
            #>
            if ($this.moduleinfo.meanimage.status -ne 'FINISHED'){
                return 1
            }
            #
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
            #
            if ($this.moduleinfo.meanimage.status -ne 'FINISHED'){
                return 1
            }
            #
            if ($this.checklog('batchflatfield', $true)){
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
        if ($this.moduleinfo.batchflatfield.status -ne 'FINISHED'){
            return 1
        }
        #
        if ($this.moduleinfo.warpoctets.vers -match '0.0.1'){
            return 3
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
        if ($this.moduleinfo.warpoctets.status -ne 'FINISHED'){
            return 1
        }
        #
        if ($this.moduleinfo.batchwarpkeys.vers -match '0.0.1'){
            return 3
        }
        #
        if ($this.checklog('batchwarpkeys', $true)){
            return 2
        }
        #
        if (!$this.testbatchwarpkeysfiles()){
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
        if ($this.moduleinfo.batchwarpkeys.status -ne 'FINISHED'){
            return 1
        }
        #
        if ($this.moduleinfo.batchwarpfits.vers -match '0.0.1'){
            return 3
        }
        #
        if ($this.checklog('batchwarpfits', $true)){
            return 2
        }
        #
        if (!$this.testbatchwarpfitsfiles()){
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
        if ($this.moduleinfo.batchwarpfits.status -ne 'FINISHED'){
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
        $this.cleanedtasks = $this.cleanedtasks | ForEach-Object {$_.Split(',')[0..3] -join(',')}
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
    [int]checkvminform($antibody){
        #
        if ($this.moduleinfo.imagecorrection.status -ne 'FINISHED'){
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
        if ($this.vmq.checkforreadytask($this.project, 
        $this.slideid, $antibody)){
            return 2
        }
        #
        return 3
        #
    }
    <# -----------------------------------------
     checkmerge
     place holder
    ------------------------------------------
     Input: 
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkmerge(dependency)
    ----------------------------------------- #>
    [int]checkmerge(){
        #
        $this.getantibodies()
        #
        $this.antibodies | foreach-Object{
            #
            if ($this.moduleinfo.vminform.($_).status -ne 'FINISHED'){
                return 1
            }
            #
        }
        #
        if ($this.checklog('merge', $true)){
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
     place holder
    ------------------------------------------
     Input: 
     ------------------------------------------
     Output: returns 1 if dependency fails, 
     returns 2 if current module is still running,
     returns 3 if current module is complete
     ------------------------------------------
     Usage: $this.checkimageqa(dependency)
    ----------------------------------------- #>
    [int]checkimageqa(){
        #
        if ($this.moduleinfo.merge.status -ne 'FINISHED'){
            return 1
        }
        #
        $this.getantibodies()
        #
        if ($this.checknewimageqa($this.antibodies)){
            return 2
        }
        #
        # check each antibody column for
        # the slide in the qa file and 
        # return true if there is an X
        # 
        if(!$this.testimageqafiles($this.antibodies)){
            return 2
        }
        #
        return 3
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
    [int]checksegmaps(){
        #
        if ($this.moduleinfo.imageqa.status -ne 'FINISHED'){
            return 1
        } 
        #
        #
        if ($this.checklog('segmaps', $true)){
            return 2
        }
        #
        if(!$this.testsegmapsfiles()){
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
        $slides = $this.importslideids($this.mpath)
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
}