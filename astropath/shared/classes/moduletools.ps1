<# -------------------------------------------
 moduletools
 Benjamin Green, Andrew Jorquera- JHU
 Last Edit: 05.04.2022
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
    FLATWIM3 = 16
 }
 #
 class moduletools{
    #
    [array]$externaltasks
    [launchmodule]$sample
    [array]$processvars
    [string]$processloc
    [string]$vers
    [int]$flevel
    [string]$condalocation = '"' + $PSScriptRoot + '\..\..\utilities\Miniconda3"'
    [string]$funclocation
    [array]$logoutput
    [string]$pythonmodulename
    [array]$batchslides
    [switch]$all = $false
    #
    moduletools([hashtable]$task, [launchmodule]$sample){
        $this.sample = $sample
        $this.BuildProcessLocPaths($task)
        $this.vers = $this.sample.GetVersion($this.sample.mpath,
             $this.sample.module, $this.sample.project)
        $this.sample.checksoftware()
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
        if ($task.processloc -AND !($task.processloc -match '\*')){
            if ($this.sample.module -match 'batch'){
                $this.processloc = ($task.processloc + '\astropath_ws\' + 
                    $this.sample.module + '\'+$task.batchid)
            } else {
                $this.processloc = ($task.processloc + '\astropath_ws\' + 
                    $this.sample.module + '\'+$task.slideid)
            }
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
            $this.DownloadFlatwIm3s()
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
            if ($this.sample.vers -match '0.0.1'){
                $flatfieldfolder = $this.processvars[0]+'\flatfield'
                $this.sample.removedir($flatfieldfolder)
                $this.sample.CreateDirs($flatfieldfolder)
                $this.sample.copy($this.sample.batchflatfield(), $flatfieldfolder)
            }
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
            $sor = $this.sample.im3folder()
            try {
                $this.sample.copy($sor, $des, 'im3', 10)
                $im3files = @()
                $im3files += $this.sample.listfiles($des, '*')
                $misnamedfiles = $im3files -cmatch '.IM3'
                foreach ($file in $misnamedfiles) {
                    $newfilename = (Split-Path $file -Leaf) -replace 'IM3', 'im3'
                    Rename-Item $file $newfilename
                }
                if(!(((get-childitem ($sor+'\*') -Include '*im3').Count) -eq (get-childitem $des).count)){
                    Throw 'im3s did not download correctly'
                }
            } catch {
                $this.silentcleanup()
                Throw $_.Exception
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
            try {
                $this.sample.copy($sor, $des, 'xml', 20)
                if(!(((get-childitem ($sor+'\*') -Include '*xml').Count) -eq (get-childitem $des).count)){
                    Throw 'xmls did not download correctly'
                }
            } catch {
                $this.silentcleanup()
                Throw $_.Exception
            }
        }
        #
    }
    <# -----------------------------------------
     DownloadFlatwIm3s
     download the flatw IM3 files to the processing
     dir
     ------------------------------------------
     Usage: $this.DownloadFlatwIm3s()
    ----------------------------------------- #>
    [void]DownloadFlatwIm3s(){
        #
        if (($this.flevel -band [FileDownloads]::FLATWIM3) -eq 
            [FileDownloads]::FLATWIM3){
            $des = $this.processvars[0] +'\'+$this.sample.slideid+'\im3\flatw'
            $sor = $this.sample.flatwim3folder()
            try {
                $this.sample.copy($sor, $des, 'im3', 30)
                $flatwfiles = @()
                $flatwfiles += $this.sample.listfiles($des, '*')
                $misnamedfiles = $flatwfiles -cmatch '.IM3'
                foreach ($file in $misnamedfiles) {
                    $newfilename = (Split-Path $file -Leaf) -replace 'IM3', 'im3'
                    Rename-Item $file $newfilename
                }
                #if ($this.getcount('flatwim3', $true) -eq (get-childitem $des).count))
                if(!(((get-childitem ($sor+'\*') -Include '*im3').Count) -eq (get-childitem $des).count)){
                    Throw 'flatw im3s did not download correctly'
                }
            } catch {
                $this.silentcleanup()
                Throw $_.Exception
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
     ShredDat
        Extract data.dat files
     ------------------------------------------
     Usage: $this.ShredDat()
    ----------------------------------------- #>
    [void]ShredDat($slideid, [array]$images){
        $this.ConvertPath('shreddat', $slideid, $images)
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
        try {
            if ($type -match 'inject'){
                ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                    $this.sample.slideid -inject -verbose 4>&1 >> $externallog
            } elseif($type -match 'shreddat') {
                ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                    $this.sample.slideid -shred -dat -verbose 4>&1 >> $externallog
            } elseif($type -match 'shredxml') {
                ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                    $this.sample.slideid -shred -xml -verbose 4>&1 >> $externallog
            } 
        } catch {
            $this.silentcleanup()
            Throw $_.Exception
        }
        #
        $this.parseconvertpathlog($externallog, $type)
        #
    }
    #
    [void]ConvertPath($type, [array]$images){
        $this.sample.info(($type + " data started"))
        $externallog = $this.ProcessLog(('convertim3pathlog' + $type))
        try {
            if ($type -match 'inject'){
                ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                    $this.sample.slideid -images:$images -inject -verbose 4>&1 >> $externallog
            } elseif($type -match 'shreddat') {
                ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                    $this.sample.slideid -images:$images -shred -dat -verbose 4>&1 >> $externallog
            } elseif($type -match 'shredxml') {
                ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                    $this.sample.slideid -images:$images -shred -xml -verbose 4>&1 >> $externallog
            } 
        } catch {
            $this.silentcleanup()
            Throw $_.Exception
        }
        #
        $this.parseconvertpathlog($externallog, $type)
        #
    }
    #
    [void]ConvertPath($type, $slideid, [array]$images){
        try {
            $this.sample.info(($type + " data started"))
            $externallog = $this.ProcessLog(('convertim3pathlog' + $type))
            if ($type -match 'inject'){
                ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                    $slideid -images:$images -inject -verbose 4>&1 >> $externallog
            } elseif($type -match 'shreddat') {
                ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                    $slideid -images:$images -shred -dat -verbose 4>&1 >> $externallog
            } elseif($type -match 'shredxml') {
                ConvertIM3Path $this.processvars[0] $this.processvars[1] `
                    $slideid -images:$images -shred -xml -verbose 4>&1 >> $externallog
            } 
        } catch {
            $this.silentcleanup()
            Throw $_.Exception
        }
        #
        $this.parseconvertpathlog($externallog, $type)
        #
    }
    #   
    [void]parseconvertpathlog($externallog, $type){
        #
        $log = $this.sample.GetContent($externallog) |
        where-object  {$_ -notlike '.*' -and $_ -notlike '*PM*' -and $_ -notlike '*AM*'} | 
        foreach-object {$_.trim()}
        $this.sample.info($log)
        remove-item $externallog -force -ea Continue
        $this.sample.info(($type + " data finished"))
        #
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
        $msi = $this.sample.im3folder() +'\*'
        $m2s = Get-ChildItem $msi -include '*_M*.im3' -Exclude '*].im3'
        $errors = $m2s | ForEach-Object {($_.Name -split ']')[0] + ']'}
        #
        $errors | Select-Object -Unique | ForEach-Object {
            $ms = (Get-ChildItem $msi -filter ($_ + '*')).Name
            $mnums = $ms | ForEach-Object {[regex]::match($_,']_M(.*?).im3').groups[1].value}
            $keep = $_+'_M'+($mnums | Measure-Object -maximum).Maximum+'.im3'
            $ms | ForEach-Object{if($_ -ne $keep){remove-item -literalpath ($wd+'\'+$_) -force}}
            rename-item -literalpath ($wd+'\'+$keep) ($_+'.im3')
        }
        #
    }
    <# -----------------------------------------
     fixSIDs
     Fix all filenames that were created due to an error.
     In these cases the file names were 
     not correctly changed from the original .im3 file
     to the slideid
     ------------------------------------------
     Usage: $this.fixSIDs()
    ----------------------------------------- #>
    [void]fixSIDs(){
        #
        $this.sample.info("Fix M# files")
        $msi = $this.sample.im3folder() +'\*'
        $slideid = $this.sample.slideid + '_*.im3'
        $sampleidim3s = Get-ChildItem $msi -exclude $slideid  -include '*.im3'
        #
        if ($sampleidim3s){
            $this.sample.warning(('found '+$sampleidim3s.count+
                ' slides that appear to have scan ids not apids'))
            $sampleidim3s | foreach-object{
                $newname = $this.sample.slideid + '_[' + ($_.name -split "_\[")[1]
                $this.checkfile($_.fullname, $newname)
                #
            }
        }
        #
    }
    #
    [void]fixmlids(){
        #
        $xml = $this.sample.xmlfolder() +'\*'
        $xmlid = $this.sample.slideid + '*.xml'
        $sampleidxmls = Get-ChildItem $xml -exclude $xmlid  -include '*.xml'
        #
        if ($sampleidxmls){
            $this.sample.warning(('found '+$sampleidxmls.count+
                ' xmls that appear to have scan ids not apids'))
            $sampleidxmls | foreach-object{
                $newname = $this.sample.slideid + '_[' + ($_.name -split "_\[")[1]
                $this.checkfile($_.fullname, $newname)
                #
            }
        }
        #
    }
    #
    [void]checkapiddef(){
        #
        $this.sample.importslideids()
        $row = $this.sample.slide_data | Where-Object {
            $_.slideid -contains $this.sample.slideid
        } 
        #
        if ($row.scan -ne $this.sample.scannumber()){
            $row.scan = $this.sample.scannumber()
            $this.sample.writecsv(
                $this.sample.slide_fullfile($this.sample.mpath),
                $this.sample.slide_data
            )
        }
        #
        $this.sample.slide_local_file += '_' + $this.sample.project + '.csv'
        $this.sample.importslideids_local()
        $row = $this.sample.slide_local_data | Where-Object {
            $_.slideid -contains $this.sample.slideid
        }
        #
        #
        if(!$this.sample.slide_local_data) {
            $this.sample.slide_local_data  = @()
        }
        #
        if (!$row){
            [array]$this.sample.slide_local_data +=  $row
            $this.sample.writecsv(
                $this.sample.slide_local_fullfile($this.sample.basepath),
                $this.sample.slide_local_data
            )
        #
        } elseif ($row.scan -ne $this.sample.scannumber()){
            $row.scan = $this.sample.scannumber()
            $this.sample.writecsv(
                $this.sample.slide_local_fullfile($this.sample.basepath),
                $this.sample.slide_local_data
            )
        }
        #
    }
    #
    [void]checksampledef(){
        #
        $this.sample.importsampledef_local()
        $row = $this.sample.sampledef_local_data | Where-Object {
            $_.slideid -contains $this.sample.slideid
        }
        #
        if(!$this.sample.sampledef_local_data) {
            $this.sample.sampledef_local_data  = @()
        }
        #
        if (!$row){
            [array]$this.sample.sampledef_local_data += [PSCustomObject]@{
                SampleID = 0
                SlideID = $this.sample.slideid
                Project = $this.sample.project
                Cohort = $this.sample.cohort
                Scan = $this.sample.scannumber()
                BatchID = $this.sample.batchid
                isGood = 1
            }
            $this.sample.writecsv(
                $this.sample.sampledef_local_fullfile($this.sample.basepath),
                $this.sample.sampledef_local_data
            )
        #
        } elseif ($row.scan -ne $this.sample.scannumber()){
            $row.scan = $this.sample.scannumber()
            $this.sample.writecsv(
                $this.sample.sampledef_local_fullfile($this.sample.basepath),
                $this.sample.sampledef_local_data
            )
        }
        #
    }
    #
    [void]checkfile($file1, $filename2){
        #
        $path = Split-Path $file1
        $file2 = $path + '\' + $filename2
        #
        if (!(test-path -LiteralPath $file2)){
            $warn = 'attempting to rename', $file1, 'to', $filename2 -join ' '
            $this.sample.warning($warn)
            #
            rename-item $file1 $filename2 -EA stop
            #
        } else {
            $this.sample.warning('scan and apid files exist for the file:' + $file1)
            #
            $byte1 = (Get-Item -LiteralPath $file1).Length
            $byte2 = (Get-Item -LiteralPath $file2).Length
            $hash1 = $this.sample.FileHasher($file1, 7, $true)
            $hash2 = $this.sample.FileHasher($file2, 7, $true)
            #
            if ($byte1 -eq 0kb -and $byte2 -eq 0kb){
                $this.sample.error('Both files have 0 bytes and appear to be empty, exiting')
            } elseif ($byte1 -eq 0kb){
                $this.sample.warning('File is empty' + $file1)
                $this.sample.warning('Attempting to delete file')
                $this.sample.removefile($file1)
            } elseif ($byte2 -eq 0kb){
                $this.sample.warning('File is empty' + $file2)
                $this.sample.warning('Attempting to replace with:' + $file1)
                $this.sample.removefile($file2)
                rename-item $file1 $filename2 -EA stop
            } elseif ($byte1 -gt $byte2){
                $this.sample.warning('File with scan id is large there may have been an error in transfer')
                $this.sample.warning('Will attempt to replace apid with scan id')
                $this.sample.removefile($file2)
                rename-item $file1 $filename2 -EA stop
            } elseif ($hash1.Value -eq $hash2.Value){
                $this.sample.warning('the hash values are identical... deleting the scan id file and keeping apid file')
                $this.sample.removefile($file1)
            }
            #
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
    <# -----------------------------------------
     buildpyopts
        some of the python options are used in
        multiple python commands. Add 
        them in a string that can be easily 
        recieved when building the python
        task.
     ------------------------------------------
     Usage: $this.buildpyopts()
    ----------------------------------------- #>
    [string]buildpyopts(){
        $str = '--allow-local-edits --skip-start-finish --job-lock-timeout 0:5:0',
            $this.pyoptsnoaxquiredannos() -join ''
        return $str
    }
    #
    [string]buildpyoptscustomannotation($annotationpath){
        $str = '--allow-local-edits --skip-start-finish --job-lock-timeout 0:5:0',
            $this.pyoptsnoaxquiredannoscustomannotation($annotationpath) -join ''
        return $str
    }
    #
    [string]buildpyopts($opt){
        $str = '--allow-local-edits --use-apiddef --job-lock-timeout 0:5:0',
            $this.pyoptsnoaxquiredannos() -join ''
        return $str
    }
    #
    [string]gpuopt(){
        if (!$this.sample.isWindows()) {
            return '--noGPU'
        }
        $gpu = Get-WmiObject win32_VideoController
        if (($gpu.Name.count) -gt 1 -or 
            ($gpu.name -match 'NVIDIA') -or 
            ($gpu.name -match 'AMD')
        ){
            return ''
        } else {
            return '--noGPU'
        }
    }
    #
    [string]pyoptsnoaxquiredannos(){
        #
        $str = ''
        #
        if ($this.sample.batchid -contains $this.sample.slideid) {    
            #
            $this.batchslides | foreach-object {
                $this.sample.slideid = $_
                $str = $this.pyoptsnoaxquiredannossub()
                if ($str){
                    $this.sample.slideid = $this.sample.batchid
                    return $str
                }
            }
            #
            $this.sample.slideid = $this.sample.batchid
            #
        } else {
           $str = $this.pyoptsnoaxquiredannossub()
        }
        #
        return $str
        #
    }
    #
    [string]pyoptsnoaxquiredannoscustomannotation($annotationpath){
        #
        $str = ''
        #
        if ($this.sample.batchid -contains $this.sample.slideid) {    
            #
            $this.batchslides | foreach-object {
                $this.sample.slideid = $_
                $str = $this.pyoptsnoaxquiredannossubcustomannotation($annotationpath)
                if ($str){
                    $this.sample.slideid = $this.sample.batchid
                    return $str
                }
            }
            #
            $this.sample.slideid = $this.sample.batchid
            #
        } else {
           $str = $this.pyoptsnoaxquiredannossubcustomannotation($annotationpath)
        }
        #
        return $str
        #
    }
    #
    [string]pyoptsnoaxquiredannossub(){
        #
        $str = ''
        #
        if (test-path $this.sample.annotationxml()){
            $xmlfile = $this.sample.getcontent($this.sample.annotationxml())
            if ([regex]::Escape($xmlfile) -notmatch 'Acquired'){
                $this.sample.warning('No "Acquired" Fields in annotation xmls, including "Flagged for Acquisition" Fields.')
                $this.sample.warning('Note some fields may have failed but this cannot be determined from xml file!')
                $str = ' --include-hpfs-flagged-for-acquisition'
            }
        }
        #
        return $str
        #
    }
    #
    [string]pyoptsnoaxquiredannossubcustomannotation($annotationpath){
        #
        $str = ''
        #
        if (test-path $annotationpath){
            $xmlfile = $this.sample.getcontent($annotationpath)
            if ([regex]::Escape($xmlfile) -notmatch 'Acquired'){
                $this.sample.warning('No "Acquired" Fields in annotation xmls, including "Flagged for Acquisition" Fields.')
                $this.sample.warning('Note some fields may have failed but this cannot be determined from xml file!')
                $str = ' --include-hpfs-flagged-for-acquisition'
            }
        }
        #
        return $str
        #
    }
    #
    [void]runpythontask($taskname, $pythontask){
        #
        $externallog = $this.ProcessLog($taskname)
        $pythontask = $this.sample.CrossPlatformPaths($pythontask)
        $this.sample.info(('python task: ' + $pythontask))
        if ($this.sample.isWindows()){
            $this.sample.checkconda()
            conda activate $this.sample.pyenv()
            Invoke-Expression $pythontask *>> $externallog
            conda deactivate 
        } else{
            Invoke-Expression $pythontask *>> $externallog
        }
        $this.getexternallogs($externallog)
        #
    }
    #
    [void]runpythontask($taskname, $pythontask, $nolog){
        #
        $externallog = $this.ProcessLog($taskname)
        $pythontask = $this.sample.CrossPlatformPaths($pythontask)
        $this.sample.info(('python task: ' + $pythontask))
        if ($this.sample.isWindows()){
            $this.sample.checkconda()
            conda activate $this.sample.pyenv()
            Invoke-Expression $pythontask *>> $externallog
            conda deactivate 
        } else{
            Invoke-Expression $pythontask *>> $externallog
        }
        #
    }
    #
    [void]getexternallogs($externallog){
        #
        $this.logoutput = $this.sample.GetContent($externallog)
        $this.checkexternalerrors()
        $this.checkastropathlog()
        #
    }
    <# -----------------------------------------
     checkexternalerrors
        check if there were external errors
        when launching python. These errors
        could be in the input, that 
        sample wasn't set up correctly,
        that the dependencies aren't correct.
        if the task started correctly, (detected
        by the correct astropath formatting in the 
        log message) external task names don't match
        our module name, we have to parse the 
        external and forward the messages to the
        astropath sample logs. 
     ------------------------------------------
     Usage: $this.checkexternalerrors()
    ----------------------------------------- #>
    [void]checkexternalerrors(){
        #
        if ($this.vers -match '0.0.1' -or $this.sample.module -match 'merge'){
            $test = 'ERROR'
            if ($this.logoutput -match $test){
                $this.silentcleanup()
                $potentialerrors = ($this.logoutput -notmatch 'Exit Status' -ne '').replace(';', '')
                $this.sample.error($potentialerrors)
                Throw 'Error in matlab task'
            } elseif ($this.logoutput) {
                $this.sample.info($this.logoutput.trim())
            }
            #
        } else {
            if ($this.sample.module -match 'batchmicomp'){
                $test = $this.pythonmodulename + ' : '
            }else {
                $test = ($this.sample.project, ';', $this.sample.cohort -join '')
            }
            if ($this.logoutput -and $this.logoutput[0] -notmatch $test) {
                $this.silentcleanup()
                $potentialerrors = $this.logoutput -ne ''
                $this.sample.error($potentialerrors)
                Throw 'Error in launching python task'
            }
            #
            if ($this.sample.module -match 'warpoctets|imagecorrection'){
                $this.parsepysamplelog()
            }
            #
            if ($this.pythonmodulename -match 'cohort' ){
                if ( 
                    $this.sample.module -notmatch 'batch'    
                ){
                    $this.parsepycohortlog()
                } else {
                    $this.parsepycohortbatchlog()
                }
            }
        }
        #
    }
    <# -----------------------------------------
     parsepysamplelog
        parsepysamplelog
     ------------------------------------------
     Usage: $this.parsepysamplelog()
    ----------------------------------------- #>
    [void]parsepysamplelog(){
        $sampleoutput = $this.logoutput -match (';'+ $this.sample.slideid+';')
        $sampleoutput | ForEach-Object {
            #
            $mess = ($_ -split ';')[3]
            #
            if ($mess -notmatch 'DEBUG:'){
                $this.sample.message = $mess
                $this.sample.Writelog(2)
            }
            #
            if ($mess -match 'ERROR:'){
                $this.sample.message = $mess
                $this.sample.Writelog(4)
            }
            #
        }
    }
    <# -----------------------------------------
     parsepycohortlog
        parsepycohortlog
     ------------------------------------------
     Usage: $this.parsepycohortlog()
    ----------------------------------------- #>
    [void]parsepycohortlog(){
        $sampleoutput = $this.logoutput -match (';'+ $this.sample.slideid+';')
        if ($sampleoutput -match 'Error'){
            Throw 'Python tasked launched but there was an ERROR'
        }
    }
    <# -----------------------------------------
     parsepycohortbatchlog
        parsepycohortbatchlog
     ------------------------------------------
     Usage: $this.parsepycohortbatchlog()
    ----------------------------------------- #>
    [void]parsepycohortbatchlog(){
        $this.logoutput | ForEach-Object{
            $cslide = ($_ -split ';')[2] 
            $mess = ($_ -split ';')[3]
            if (
                ($this.batchslides -match $cslide -and
                    $mess -notmatch 'DEBUG:' -and
                    $mess -notmatch 'FINISH:' -and
                    $mess -notmatch 'START:') -or
                ($cslide -match 'project'-and
                    $mess -notmatch 'DEBUG:' -and
                    $mess -notmatch 'FINISH:' -and
                    $mess -notmatch 'START:')
            ){
                #
                if (!$mess){
                    $this.sample.error($_)                    
                } else {
                    $this.sample.message = $mess
                    $this.sample.Writelog(4)
                }
                #
            }
        }
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
            $ID= $this.sample.BatchID.padleft(2,'0')
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
                    where-object {($_.Slideid -contains $ID) -and 
                        ($_.Message -match "^$statustype")} |
                    Select-Object -Last 1 
        }
        #
        $d1 = ($savelog | Where-Object {$_.Message -match $statustypes[0]}).Date
        $d2 = ($savelog | Where-Object {$_.Message -match $statustypes[1]}).Date
        #
        if (($d2) -and ($d2 -gt $d1)){
            $this.silentcleanup()
            Throw 'detected error in external task'
        }
        #
    
    }
    #
    [void]getslideidregex(){
        #
        if ($this.all){
            $this.sample.importslideids($this.sample.mpath)
            $aslides = $this.sample.slide_data |
                where-object {$_.Project -contains $this.sample.project}
            $slides = $aslides.SlideID
        } else {
            $slides = $this.sample.batchslides.slideid
        }
        #
        $this.batchslides = $slides
        #
    }
    #
    [void]getslideidregex($cmodule){
        #
        $this.sample.info('selecting samples for sample regex')
        #
        $nbatchslides = @()
        $sid = $this.sample.slideid
        #
        $this.getslideidregex()
        #
        if (@('batchwarpkeys', 'batchwarpfits') -match $cmodule){
            foreach ($slide in $this.batchslides){
                $this.sample.slideid = $slide
                if ($this.sample.testwarpoctetsfiles()){
                    $nbatchslides += $slide
                }
            }
        } else {
            $nbatchslides = $this.batchslides
        }
        #
        $this.sample.slideid = $sid
        $this.sample.info(([string]$nbatchslides.length +
                ' sample(s) selected for sample regex'))
        $this.batchslides = $nbatchslides
        #
    }
    #
 }
