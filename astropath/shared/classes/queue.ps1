<# -------------------------------------------
 queue
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build the queues and check 
 dependencies
 -------------------------------------------#>
class queue : sharedtools{
    #
    [Array]$originaltasks
    [Array]$cleanedtasks
    [string]$queue_file
    [string]$informvers
    [string]$project
    #
    queue($mpath, $module){
        $this.mpath = $mpath
        $this.module = $module 
    }
    queue($mpath, $module, $project){
        $this.mpath = $mpath
        $this.module = $module 
        $this.project = $project
    }
    queue($mpath, $module, $project, $slideid){
        $this.mpath = $mpath
        $this.module = $module 
        $this.project = $project
        $this.slideid = $slideid.trim()
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
            $this.updateCSVFiles()
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
        $project_dat = $this.ImportConfigInfo($this.mpath)
        #
        # select samples from the appropriate modules 
        #
        if ($this.project -eq $null){
            $projects = ($project_dat | 
                Where-object {$_.($this.module) -match 'yes'}).Project
        } else {
            $projects = $this.project
        }
        #
        $cleanedslides = $slides | 
            Where-Object {$projects -contains $_.Project}
        #
        $slidesnotcomplete = $this.defNotCompletedSlides($cleanedslides)
        $slidearray = @()
        $batcharray = @()
        if ($slidesnotcomplete.count -eq 1){
            $slidearray += $slidesnotcomplete.Project + 
                ',' + $slidesnotcomplete.Slideid
            $batcharray += $slidesnotcomplete.Project + 
                ',' + $slidesnotcomplete.Slideid
        } else {
            for($i=0; $i -lt $slidesnotcomplete.count;$i++){
                $slidearray += $slidesnotcomplete.Project[$i] + 
                    ',' + $slidesnotcomplete.Slideid[$i]
                $batcharray += $slidesnotcomplete.Project[$i] + 
                    ',' + $slidesnotcomplete.BatchID[$i]
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
     updateCSVFiles
     update csv files
     ------------------------------------------
     Usage: $this.updateCSVFiles()
    ----------------------------------------- #>
    [void]updateCSVFiles(){
        
        $localqueuelocation = '\\bki08\h$\testing\upkeep_and_progress\inForm_queue.csv'
        $mainqueuelocation = '\\bki08\h$\testing\astropath_processing\across_project_queues\vminform-queue - Prototype.csv'
        $localqueue = $this.GetContent($localqueuelocation)
        $mainqueue = $this.GetContent($mainqueuelocation)
        #
        $localcopy = ''
        foreach($row in $localqueue){
            $array = $row.ToString().split(',')
            $array = $array -replace '\s', ''
            $localcopy += $array

        }

        foreach($row in $mainqueue){
            $array = $row.ToString().split(',')
            $array = $array -replace '\s', ''
            if (!($array[0] -match 'T'+$this.project.PadLeft(3,'0'))){continue}
            $projectid = $array[0].substring(4).trim('0')


        }

        #
        #$this.informvers = '2.4.8'
        #
        #$queue_path = $this.mpath + '\across_project_queues'
        #$this.queue_file = $queue_path + '\' + $this.module + '-queue.csv'
        #$queue_data = $this.getcontent($this.queue_file)
        #
        #$current_queue_data = @()
        #
        # find rows without "processing started"
        #
        #foreach($row in $queue_data) {
        #    $array = $row.ToString().Split(",")
        #    $array = $array -replace '\s',''
        #    if($array[3]){
        #        if($array -match "Processing"){ Continue } else { 
        #            $current_queue_data += $row
        #            }
        #        } 
        #}
        #
        #$this.originaltasks = $current_queue_data
        #$this.cleanedtasks = $this.originaltasks -replace ('\s','')
        #$this.cleanedtasks = $this.cleanedtasks | ForEach {$_.Split(',')[0..3] -join(',')}
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
        $c = 1
        $ctotal = $cleanedslides.count
        #
        foreach($slide in $cleanedslides){
            #
            $p = [math]::Round(100 * ($c / $ctotal))
            Write-Progress -Activity "Checking slides" `
                           -Status "$p% Complete:" `
                           -PercentComplete $p `
                           -CurrentOperation $slide.slideid
            $c += 1 
            #
            $log = [mylogger]::new($this.mpath, $this.module, $slide.slideid)
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
        $loglines = $this.opencsvfile($log.slidelog, ';', @('Project','Cohort','slideid','Message','Date'))
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
    [int]checkmeanimagecomparison([mylogger]$log, $dependency){
        #
        if (!($this.checkmeanimage($log, $true) -eq 3)){
            return 1
        }
        #
        try {
            $cvers = $this.getversion($this.mpath, 'meanimagecomparison', $log.project)
            #if slide is not in the meanimagecomparison file return 2
        } catch {}
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
        if (!($this.checkmeanimage($log, $true) -eq 3)){
            return 1
        }
        #
        $log = [mylogger]::new($this.mpath, 'batchflatfield', $log.slideid)
        $log.slidelog = $log.mainlog
        if ($this.checklog($log, $true)){
            return 2
        }
        #
        # version depedendent checks
        #
        if (!$log.testbatchflatfield()){
            if ($log.vers -match '0.0.1'){
                return 2
            } else {
                return 1
            }
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
    [array]Aggregatebatches($batcharray){
        $batcharrayunique = $batcharray | Sort-Object | Get-Unique
        $slides = $this.importslideids($this.mpath)
        $batchescomplete = @()
        #
        $batcharrayunique | foreach-object {
            $nslidescomplete = ($batcharray -match $_).count
            $projectbatchpair = $_ -split ','
            $sample = [sampledef]::new($this.mpath, $this.module, $projectbatchpair[1], $projectbatchpair[0])
            $nslidesbatch = $sample.batchslides.count
            if ($nslidescomplete -eq $nslidesbatch){
                $batchescomplete += $_
            }
        }
        return $batchescomplete
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