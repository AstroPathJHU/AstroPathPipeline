<# -------------------------------------------
 moduletools
 created by: Benjamin Green - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 general functions which may be needed by
 multiple modules
 -------------------------------------------#>
  [Flags()] Enum FileDownloads {
    IM3 = 1
    FLATFIELD = 2
    BATCHID = 4
    XML = 8
 }
 #
 Class moduletools{
    #
    [array]$externaltasks
    [launchmodule]$sample
    [array]$processvars
    [string]$processloc
    [string]$vers
    [int]$flevel
    [string]$condalocation = '"' + $PSScriptRoot + '\..\..\utilities\Miniconda3"'
    [string]$funclocation
    [arraypower]$logoutput
    [string]$pythonmodulename
    #
    moduletools([array]$task,[launchmodule]$sample){
        $this.sample = $sample
        $this.BuildProcessLocPaths($task)
        $this.vers = $this.sample.GetVersion(
            $this.sample.mpath, $this.sample.module, $task[0])   
    }
    <# -----------------------------------------
    BuildProcessLocPath
    build the processing specimens directory paths
    if it does not exist. If input is '*' then
    work 'in place'
    ------------------------------------------
    Usage: $this.BuildProcessLoc()
    ----------------------------------------- #>
    #
    [void]BuildProcessLocPaths($task){
        $fwpath = '\\'+$this.sample.project_data.fwpath
        $this.processvars = @($this.sample.basepath, $fwpath, `
            $this.sample.flatwim3folder(), $this.sample.batchflatfield())
        #
        # If processloc is not '*' a processing destination was added as 
        # input, correct the paths to analyze from there
        #
        if ($task[2] -AND !($task[2] -match '\*')){
            $this.processloc = ($task[2] + '\astropath_ws\' + 
                $this.sample.module + '\'+$task[1])
            #
            $processvarsa = $this.processvars[0,2,3] -replace `
                [regex]::escape($this.sample.basepath), $this.processloc 
            $processvarsb = $this.processvars[1] -replace `
                [regex]::escape('\\'+$this.sample.project_data.fwpath), `
                ($this.processloc+'\flatw')
            $this.processvars = @($processvarsa[0], $processvarsb, `
                $processvarsa[1], $processvarsa[2], 1)
        } else {
            $this.processloc = $this.sample.flatwfolder()
        }
    }
    <# -----------------------------------------
     ProcessLog
     build log location strings for different
     external processes
     ------------------------------------------
     Usage: $this.processlog($processloc, $task)
    ----------------------------------------- #>
    [string]ProcessLog($externaltask){
        $out = $this.processloc + '\' + $externaltask + '.log'
        return $out
    }
    <# -----------------------------------------
     DownloadIm3
     Download the im3s to process; reduces network
     strain and frequent network errors while 
     processing
     ------------------------------------------
     Usage: $this.DownloadIm3($im3, $flatfield, $batchid)
    ----------------------------------------- #>
    [void]DownloadFiles(){
        if ($this.processvars[4]){
            $this.sample.info("Download Files started")
            $this.WipeProcessDirs()
            $this.BuildProcessDirs()
            $this.Downloadflatfield()
            $this.DownloadIm3s()
            $this.DownloadBatchID()
            $this.DownloadXML()
            $this.sample.info("Download Files finished")
        }
    }
    <# -----------------------------------------
     WipeProcessDirs
     wipe the processing directory
     ------------------------------------------
     Usage: $this.WipeProcessDirs()
    ----------------------------------------- #>
    [void]WipeProcessDirs(){
        #
        foreach($ii in @(0,1,2)){
            $this.sample.removedir($this.processvars[$ii])
        }
        #
    }
    <# -----------------------------------------
     BuildProcessDirs
     Build the processing directory
     ------------------------------------------
     Usage: $this.BuildProcessDirs()
    ----------------------------------------- #>
    [void]BuildProcessDirs(){
        #
        foreach($ii in @(0,1,2)){
            $this.sample.CreateDirs($this.processvars[$ii])
        }
        #
    }
    <# -----------------------------------------
     Downloadflatfield
     download the flatfield file to the processing
     dir
     ------------------------------------------
     Usage: $this.Downloadflatfield()
    ----------------------------------------- #>
    [void]Downloadflatfield(){
        #
        if (($this.flevel -band [FileDownloads]::FLATFIELD) -eq 
            [FileDownloads]::FLATFIELD){
            $flatfieldfolder = $this.processvars[0]+'\flatfield'
            $this.sample.removedir($flatfieldfolder)
            $this.sample.CreateDirs($flatfieldfolder)
            $this.sample.copy($this.sample.batchflatfield(), $flatfieldfolder)
        }
        #
    }
    <# -----------------------------------------
     DownloadIm3s
     download the IM3 files to the processing
     dir
     ------------------------------------------
     Usage: $this.DownloadIm3s()
    ----------------------------------------- #>
    [void]DownloadIm3s(){
        #
        if (($this.flevel -band [FileDownloads]::IM3) -eq 
            [FileDownloads]::IM3){
            $des = $this.processvars[0] +'\'+
                $this.sample.slideid+'\im3\'+$this.sample.Scan()+,'\MSI'
            $sor = $this.sample.MSIfolder()
            $this.sample.copy($sor, $des, 'im3', 30)
            if(!(((gci ($sor+'\*') -Include '*im3').Count) -eq (gci $des).count)){
                Throw 'im3s did not download correctly'
            }
        }
        #
    }
    <# -----------------------------------------
     DowloadBatchID
     download the batch id file to the processing
     dir
     ------------------------------------------
     Usage: $this.DowloadBatchID()
    ----------------------------------------- #>
    [void]DownloadBatchID(){
        #
        if (($this.flevel -band [FileDownloads]::BATCHID) -eq
             [FileDownloads]::BATCHID){
            $des = $this.processvars[0] +'\'+
                $this.sample.slideid+'\im3\'+$this.sample.Scan()
            $this.sample.copy($this.sample.BatchIDfile(), $des)
        }
        #
    }
    <# -----------------------------------------
     DowloadXML
     download the xml files to the processing
     dir
     ------------------------------------------
     Usage: $this.DowloadXML()
    ----------------------------------------- #>
    [void]DownloadXML(){
        #
        if (($this.flevel -band [FileDownloads]::XML) -eq 
            [FileDownloads]::XML){
            $des = $this.processvars[1] +'\' + $this.sample.slideid + '\'
            $sor = $this.sample.xmlfolder()
            $this.sample.copy($sor, $des, 'xml', 30)
            if(!(((gci ($sor+'\*') -Include '*xml').Count) -eq (gci $des).count)){
                Throw 'xmls did not download correctly'
            }
        }
        #
    }
    <# -----------------------------------------
     ShredDat
        Extract data.dat files
     ------------------------------------------
     Usage: $this.ShredDat()
    ----------------------------------------- #>
    [void]ShredDat(){
        $this.ConvertPath('shreddat')
    }
    <# -----------------------------------------
     ShredXML
        Extract xml files
     ------------------------------------------
     Usage: $this.ShredDat()
    ----------------------------------------- #>
    [void]ShredXML(){
        $this.ConvertPath('shredxml')
    }
    <# -----------------------------------------
     InjectDat
        inject the data from the Data.dat files
        back into the im3s and put im3s into
        flatwim3 location 
     ------------------------------------------
     Usage: $this.InjectDat()
    ----------------------------------------- #>
    [void]InjectDat(){
        $this.ConvertPath('inject')
    }
    <# -----------------------------------------
     ConvertPath
        run convert path with inject or shred
        as input above
     ------------------------------------------
     Usage: $this.ConvertPath()
    ----------------------------------------- #>
    [void]ConvertPath($type){
        $this.sample.info(($type + " data started"))
        $externallog = $this.ProcessLog(('convertim3pathlog' + $type))
        if ($type -match 'inject'){
            ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                $this.sample.slideid -i -verbose 4>&1 >> $externallog
        } elseif($type -match 'shreddat') {
            ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                $this.sample.slideid -s -d -verbose 4>&1 >> $externallog
        } elseif($type -match 'shredxml') {
            ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                $this.sample.slideid -s -xml -verbose 4>&1 >> $externallog
        } 
        $log = $this.sample.GetContent($externallog) |
             where-object  {$_ -notlike '.*' -and $_ -notlike '*PM*' -and $_ -notlike '*AM*'} | 
             foreach {$_.trim()}
        $this.sample.info($log)
        remove-item $externallog -force -ea Continue
        $this.sample.info(($type + " data finished"))
    }
    <# -----------------------------------------
     fixM2
     Fix all filenames that were created due to an error.
     In these cases the original .im3 file has been truncated,
     it exists but cannot be used. The Vectra system then
     re-wrote the file, but padded the filename with _M2.
     Here we do two things: if there is an _M2 file, we first
     delete the file with the short length, then rename the file.
     ------------------------------------------
     Usage: $this.fixM2()
    ----------------------------------------- #>
    [void]fixM2(){
        #
        $this.sample.info("Fix M# files")
        $msi = $this.sample.MSIfolder() +'\*'
        $m2s = gci $msi -include '*_M*.im3' -Exclude '*].im3'
        $errors = $m2s | ForEach-Object {($_.Name -split ']')[0] + ']'}
        #
        $errors | Select-Object -Unique | ForEach-Object {
            $ms = (gci $msi -filter ($_ + '*')).Name
            $mnums = $ms | ForEach-Object {[regex]::match($_,']_M(.*?).im3').groups[1].value}
            $keep = $_+'_M'+($mnums | Measure -maximum).Maximum+'.im3'
            $ms | ForEach-Object{if($_ -ne $keep){remove-item -literalpath ($wd+'\'+$_) -force}}
            rename-item -literalpath ($wd+'\'+$keep) ($_+'.im3')
        }
        #
    }
    #
    [void]runmatlabtask($taskname, $matlabtask){
        #
        $externallog = $this.ProcessLog($taskname)
        matlab -nosplash -nodesktop -minimize -sd $this.funclocation -batch $matlabtask -wait *>> $externallog
        $this.getexternallogs($externallog)
        #
    }
    #
    [void]runpythontask($taskname, $pythontask){
        #
        $externallog = $this.ProcessLog($taskname)
        $this.sample.checkconda()
        conda activate $this.sample.pyenv()
        Invoke-Expression $pythontask *>> $externallog
        conda deactivate $this.sample.pyenv()
        $this.getexternallogs($externallog)
        #
    }
    #
    [void]getexternallogs($externallog){
        #
        $this.logoutput = $this.sample.GetContent($externallog)
        $this.sample.removefile($externallog)
        $this.checkexternalerrors()
        $this.checkastropathlog()
        #
    }
    <# -----------------------------------------
     checkexternalerrors
        checkexternalerrors
     ------------------------------------------
     Usage: $this.checkexternalerrors()
    ----------------------------------------- #>
    [void]checkexternalerrors(){
        #
        if ($this.vers -match '0.0.1'){
            $test = 'ERROR'
            if ($this.logoutput -match $test){
                $this.silentcleanup()
                $potentialerrors = ($this.logoutput.trim() -ne '') -notmatch 'ERROR'
                Throw $potentialerrors
            } elseif ($this.logoutput) {
                $this.sample.info($this.logoutput.trim())
            }
            #
        } else {
            $test = $this.pythonmodulename + ' : ' +
                $this.sample.project + ';' + $this.sample.cohort
            if ($this.logoutput[0] -notmatch $test) {
                $this.silentcleanup()
                $potentialerrors = $this.logoutput.trim() -ne ''
                Throw $potentialerrors
            }
        }
        #
    }
    <# -----------------------------------------
     checkastropathlog
        checkastropathlog
     ------------------------------------------
     Usage: $this.checkastropathlog()
    ----------------------------------------- #>
    [void]checkastropathlog(){
        #
        $loglines = import-csv $this.sample.mainlog `
            -Delimiter ';' `
            -header 'Project','Cohort','slideid','Message','Date' 
        
        #
        # parse log
        #
        if ($this.sample.module -match 'batch'){
            $ID= $this.sample.BatchID
        } else {
            $ID = $this.sample.slideid
        }
        $statustypes = @('START:','ERROR:','FINISH:')
        $savelog = @()
        $parsedvers = $this.vers -replace 'v', ''
        $parsedvers = ($parsedvers -split '\.')[0,1,2] -join '.'
        #
        foreach ($statustype in $statustypes){
            $savelog += $loglines |
                    where-object {($_.Slideid -match $ID) -and 
                        ($_.Message -match $statustype)} |
                    Select-Object -Last 1 
        }
        #
        $d1 = ($savelog | Where-Object {$_.Message -match $statustypes[0]}).Date
        $d2 = ($savelog | Where-Object {$_.Message -match $statustypes[1]}).Date
        #
        if ($d2 -gt $d1){
            $this.silentcleanup()
            Throw 'detected error in external task'
        }
        #
    }
 }