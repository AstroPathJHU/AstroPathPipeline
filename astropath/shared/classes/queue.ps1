   ##
# manage the queues 

class queue : sharedtools{
    #
    [Array]$originaltasks
    [Array]$cleanedtasks
    [string]$queue_file
    [string]$vers
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
    #
    # gets available tasks from the queue
    #
    [void]ExtractQueue(){
        #
        if ('vminform' -ne $this.module){
            $this.buildqueue()
        } else {
            $this.('check'+$this.module)()
        }
        #
    }
    #
    [void]buildqueue(){
        #
        $slides = $this.importslideids($this.mpath)
        $project_dat = $this.ImportConfigInfo($this.mpath)
        #
        # select samples from the appropriate modules 
        #
        if ($this.project -eq $null){
            $projects = ($project_dat | Where-object {$_.($this.module) -match 'yes'}).Project
        } else {
            $projects = $this.project
        }
        #
        $cleanedslides = $slides | Where-Object {$projects -contains $_.Project}
        #
        $slidesnotcomplete = $this.defNotCompletedSlides($cleanedslides)
        $slidearray = @()
        $batcharray = @()
        if ($slidesnotcomplete.count -eq 1){
            $slidearray += $slidesnotcomplete.Project + ',' + $slidesnotcomplete.Slideid
            $batcharray += $slidesnotcomplete.Project + ',' + $slidesnotcomplete.Slideid
        } else {
            for($i=0; $i -lt $slidesnotcomplete.count;$i++){
                $slidearray += $slidesnotcomplete.Project[$i] + ',' + $slidesnotcomplete.Slideid[$i]
                $batcharray += $slidesnotcomplete.Project[$i] + ',' + $slidesnotcomplete.BatchID[$i]
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
    #
    [array]defNotCompletedSlides($cleanedslides){
        $slidesnotcomplete = @()
        foreach($slide in $cleanedslides){
            #
            $log = [mylogger]::new($this.mpath, $this.module, $slide.slideid)
            #
            if($this.checklog($log)) {
                #
                if (($this.('check'+$this.module)($log, $true) -eq 2)) {
                    #
                    $slidesnotcomplete += $slide
                    #
                }
                #
            }
            #
        }
        #
        return $slidesnotcomplete
    }
    #
    # returns true if the slide has not yet been completed
    #
    [switch]checklog([mylogger]$log){
        #
        if (!(test-path $log.slidelog)){
            return $true
        }
        #
        $loglines = import-csv $log.slidelog -Delimiter ';' -header 'Project','Cohort','slideid','Message','Date' 
        #
        # parse log
        #
        $statustypes = @('Started','Error','Finished')
        $savelog = @()
        #
        foreach ($statustype in $statustypes){
            $savelog += $loglines |
                    where-object {($_.Message -match $vers) -and ($_.Slideid -match $slideid) -and ($_.Message -match $statustype)} |
                    Select-Object -Last 1 
        }
        #
        $d1 = ($savelog | Where-Object {$_.Message -match 'START'}).Date
        $d2 = ($savelog | Where-Object {$_.Message -match 'ERROR'}).Date
        $d3 = ($savelog | Where-Object {$_.Message -match 'FINISH'}).Date
        #
        if (!$d3 -or ($d1 -lt $d2 -and $d3 -ge $d2)){
            return $true
        } else { 
            return $false
        }
        #
    }
    #
    [int]checktransfer([mylogger]$log){
        #
        # check for checksum, qptiff, and annotationxml
        # 
        $file = $log.CheckSumsfile()
        $file2 = $log.qptifffile()
        $file3 = $log.annotationxml()
        $im3s = (gci ($log.Scanfolder() + '\MSI\*') *im3).Count
        #
        if (!(test-path $file)){
            return 2
        }
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
    #
    [int]checkshredxml([mylogger]$log, $dependency){
        #
        if ($dependency){
            if (!($this.checktransfer($log) -eq 3)){
                return 1
            }
        }
        #
        # check for xmls
        # 
        $xml = $log.xmlfolder()
        $im3s = (gci ($log.Scanfolder() + '\MSI\*') *im3).Count
        #
        if (!(test-path $xml)){
            return 2
        }
        #
        # check files = im3s
        #
        $files = (gci ($xml + '\*') '*xml').Count
        if (!($im3s -eq $files)){
            return 2
        }
        #
        return 3
        #
    }
    #
    [int]checkmeanimage([mylogger]$log, $dependency){
        #
        if ($dependency){
            if (!($this.checkshredxml($log, $true) -eq 3)){
                return 1
            }
        }
        #
        # check version
        #
        $cvers = $this.getversion($this.mpath, 'meanimage', $log.project)
        if ($cvers -eq '0.0.1'){
            #
            # check for mean images
            # 
            $file = $log.im3folder() + '\' + $log.slideid + '-mean.csv'
            $file2 = $log.im3folder() + '\' + $log.slideid + '-mean.flt'
            #
            if (!(test-path $file)){
                return 2
            }
            if (!(test-path $file2)){
                return 2
            }
        } else {
            #
            # check for meanimage directory
            #
            $p = $log.meanimagefolder()
            if (!(test-path $p)){
                return 2
            }
        }
        #
        return 3
        #
    }
    #
    [switch]checkmeanimagecomparison([mylogger]$log, $dependency){
        #
        if ($dependency){
           if (!($this.checkmeanimage($log, $true) -eq 3)){
                return 1
           }
        }
        #
        try {
            $cvers = $this.getversion($this.mpath, 'meanimage', $log.project)
            #if slide is not in the meanimagecomparison file return true
        } catch {}
        #
        return 3
        #
    }
    #
    [switch]checkbatchflatfield([mylogger]$log, $dependency){
        #
        if ($dependency){
           if (!($this.checkmeanimagecomparison($log, $true) -eq 3)){
                return 1
           }
        }
        # 
        $file = $log.batchflatfield()
        #
        if (!(test-path $file)){
            return 2
        }
        return 3
        #
    }
    #
    [switch]checkimagecorrection([mylogger]$log, $dependency){
        #
        if ($dependency){
            if (!($this.checkbatchflatfield($log, $true) -eq 3)){
                return 1
            }
        }
        #
        $im3s = (gci ($log.Scanfolder() + '\MSI\*') *im3).Count
        #
        $paths = @($log.flatwim3folder(), ('\\'+$log.flatwfolder()), ('\\'+$log.flatwfolder()))
        $filetypes = @('*im3', '*fw', '*fw01')
        #
        for ($i=0; $i -lt 3; $i++){
            #
            if (!(test-path $paths[$i])){
                return 2
            }
            #
            # check files = im3s
            #
            $files = (gci ($paths[$i] + '\*') $filetypes[$i]).Count
            if (!($im3s -eq $files)){
                return 2
            }
        }
        #
        return 3
    }
    #
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
    #
    # check that all slides from each unqiue batch are on the list
    # return one sample
    #
    [array]Aggregatebatches($batcharray){
        $batcharray = $batcharray | Sort-Object | Get-Unique
        return $batcharray
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