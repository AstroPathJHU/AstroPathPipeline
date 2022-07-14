<#
--------------------------------------------------------
informinput
Created By: Benjamin Green, Andrew Jorquera
Last Edit: 03/28/2022
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[hashtable]: must contain slideid, 
    antibody, algorithm, and inform version to use
    E.g. @('\\bki04\Clinical_Specimen_2','M18_1','CD8,CD8_12.05.2018_highTH.ifr','2.4.8.)
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [vminform]::new($task, $sample)
       $a.RunBatchInForm()
--------------------------------------------------------
#>
Class vminform : moduletools {
    #
    [string]$abx
    [string]$alg
    [string]$abpath
    [string]$algpath
    [string]$outpath = "C:\Users\Public\BatchProcessing"
    [string]$informoutpath
    [string]$image_list_file = $this.outpath + "\image_list.tmp"
    [array]$image_list
    [string]$informpath
    [launchmodule]$sample
    [string]$informbatchlog
    [int]$err
    [string]$informprocesserrorlog =  $this.outpath + "\informprocesserror.log"
    [array]$corruptedfiles
    [array]$skippedfiles
    [bool]$needsbinaryseg
    [bool]$needscomponent
    [string]$inputimagepath
    [array]$inputimageids
    [switch]$islocal = $true
    #
    $export_type_setting = @{
        Default          = (('     <ExportTypes>eet_NucSegmentation,',
                                      'eet_CytoSegmentation,',
                                      'eet_MembraneSegmentation</ExportTypes>') -join " ");
        BinaryMaps       = (('     <ExportTypes>eet_Segmentation,',
                                      'eet_NucSegmentation,',
                                      'eet_CytoSegmentation,',
                                      'eet_MembraneSegmentation</ExportTypes>') -join " ");
        Component        = (('     <ExportTypes>eet_NucSegmentation,',
                                      'eet_CytoSegmentation,',
                                      'eet_MembraneSegmentation,',
                                      'eet_ComponentData</ExportTypes>') -join " ");
        BinaryWComponent = (('     <ExportTypes>eet_Segmentation,',
                                      'eet_NucSegmentation,',
                                      'eet_CytoSegmentation,',
                                      'eet_MembraneSegmentation,',
                                      'eet_ComponentData</ExportTypes>') -join " ")
    }
    #
    $error_dictionary = @{
        ConnectionFailed = 'A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond';
        NoElements = 'Sequence contains no elements';
        SegmentCells = 'Please segment cells';
        CorruptIM3 = 'External component has thrown an exception';
        OverlappedObjects = 'The stencil contains overlapped objects'
    }
    #
    vminform([hashtable]$task,[launchmodule]$sample) : base ([hashtable]$task, [launchmodule]$sample) {
        #
        $this.flevel = [FileDownloads]::FLATWIM3
        #
        $this.sample = $sample
        $this.abx = $task.antibody.trim()
        $this.alg = $task.algorithm.trim()
        $this.abpath = $this.sample.phenotypefolder() + '\' + $this.abx
        $this.algpath = $this.sample.basepath +
             '\tmp_inform_data\Project_Development\' + $this.alg
        $this.informoutpath = $this.outpath + "\" + $this.abx + '_' + $this.err
        $this.informpath = '"'+"C:\Program Files\Akoya\inForm\" + 
            $task.informvers.trim() + "\inForm.exe"+'"'
        $this.informbatchlog = $this.informoutpath + "\Batch.log"
        $this.processvars[0] = $this.outpath
        $this.processvars[1] = $this.outpath
        $this.processvars[2] = $this.outpath
        #
        if ($this.islocal){
            $this.inputimagepath = $this.sample.flatwim3folder()
        } else {
            $this.inputimagepath = $this.outpath + '\' + $this.sample.slideid + '\im3\flatw'
        }
        $this.processvars += 1
        #
        $this.TestPaths()
        $this.KillinFormProcess()
        #
    }
    <# -----------------------------------------
     TestPaths
     Test that the algorithm, flatw, and inform
     version all exist in the correct locations
     ------------------------------------------
     Usage: $this.TestPaths()
    ----------------------------------------- #>
    [void]TestPaths(){
        #
        if (!(test-path $this.algpath)){
            Throw "algorithm not found for:" + $this.algpath
        }
        if (!(test-path $this.sample.flatwim3folder())){
            Throw "flatw path not found for:" + $this.sample.flatwim3folder()
        }
        #
    }
    <# -----------------------------------------
     RunVMinForm
     Run the virtual machine inform process 
     ------------------------------------------
     Usage: $this.RunVMinForm()
    ----------------------------------------- #>
    [void]RunVMinForm(){
        #
        $this.setupvmdirs()
        #
        while(($this.err -le 5) -AND ($this.err -ge 0)){
            $this.CreateOutputDir()
            $this.CreateImageList()
            $this.CheckExportOptions()
            $this.StartInForm()
            $this.WatchBatchInForm()
            $this.CheckErrors()
            if (($this.err -le 5) -and ($this.err -gt 0)){
                $this.sample.warning("Task will restart. Attempt "+ $this.err)
            } elseif ($this.err -gt 5){
                Throw "Could not complete task after 5 attempts"
            } elseif ($this.err -eq -1){
                $this.sample.info("inForm Batch Process Finished Successfully")
                $this.MergeOutputDirectories()
                $this.informoutpath = $this.outpath + "\" + $this.abx + '_0'
            }
        }
        #
        $this.ReturnData()
        #
    }
    <# -----------------------------------------
     setupvmdirs
     ------------------------------------------
     Usage: $this.setupvmdirs()
    ----------------------------------------- #>
    [void]setupvmdirs(){
        #
        $this.sample.info("Set up batch processing dir")
        $this.sample.createnewdirs($this.outpath)
        $this.sample.copy($this.algpath, $this.outpath)
        $this.algpath = $this.outpath + '\' + $this.alg
        #
        if (!$this.islocal){
            $this.DownloadFiles()
        }
        #
    }
    <# -----------------------------------------
     CreateOutputDir
     First kill any old inForm processes that are
     running on the system which would cause the 
     current processing to crash. Next, delete 
     any old processing directory and create a
     new processing directory.
     ------------------------------------------
     Usage: $this.CreateOutputDir()
    ----------------------------------------- #>
    [void]CreateOutputDir(){
        #
        $this.sample.info("Create inForm output location")
        $this.KillinFormProcess()
        $this.informoutpath = $this.outpath + "\" + $this.abx + '_' + $this.err
        $this.informbatchlog = $this.informoutpath + '\Batch.log'
        $this.sample.createnewdirs($this.informoutpath)
        #
    }
    <# -----------------------------------------
     KillinFormProcess
     Kill any inForm processes that are running
     on the system 
     ------------------------------------------
     Usage: $this.KillinFormProcess()
    ----------------------------------------- #>
    [void]KillinFormProcess(){
        #
        get-process -name inform -EA SilentlyContinue |
            Stop-Process -Force -EA stop | Out-Null
        #
        get-process |
            where-object {$_.MainWindowTitle -eq 'inform'} |
            Stop-process -force -EA Stop | out-null
        #
        get-process -name rserve -EA SilentlyContinue |
            Stop-Process -Force -EA stop | Out-Null
        Start-Sleep 20
        #
    }
    <# -----------------------------------------
     CreateImageList
     Create a list of images for inForm to process
     and save it to the processing directory
     ------------------------------------------
     Usage: $this.CreateImageList()
    ----------------------------------------- #>
    [void]CreateImageList(){
        #
        $this.sample.info("Compile image list")
        if (!$this.inputimageids){
            $this.inputimageids = (Get-ChildItem -Path ($this.inputimagepath + '\*') -include *.im3).FullName
        }
        $this.image_list =  $this.inputimageids |
            foreach-object {$_+"`r`n"}
        $this.sample.setfile($this.image_list_file, $this.image_list)
        #
    }
    <# -----------------------------------------
     CheckExportOptions
     Gets MergeConfig data and checks/updates
     the inform procedure output options to 
     toggle segmentation table data and 
     file type data
     ------------------------------------------
     Usage: $this.CheckExportOptions()
    ----------------------------------------- #>
    [void]CheckExportOptions(){
        #
        $this.GetMergeConfigData()
        #
        $procedure = $this.sample.GetContent($this.algpath)
        if ($this.abx -notmatch 'Component') {
            $procedure = $this.CheckSegmentationTableOption($procedure, 'false', 'true')
        }
        else {
            $procedure = $this.CheckSegmentationTableOption($procedure, 'true', 'false')
        }
        #
        $procedure = $this.CheckCoordinateSpaceIndexOption($procedure)
        $this.CheckExportLineOption($procedure)
        #
    } 
    <# -----------------------------------------
     CheckSegmentationTableOption
     Checks the protocol file for the 
     segmnetationtable option and sets it 
     depending on whether the procedure name
     is 'component'
     ------------------------------------------
     Usage: $this.CheckSegmentationTableOption()
    ----------------------------------------- #>
    [array]CheckSegmentationTableOption($procedure, $from, $to){
        $segmentationtableline = $procedure | 
            Where-Object {$_ -match '<SegmentationTable>'}
        if (!$segmentationtableline) {
            throw 'error in reading <SegmentationTable> line in procedure'
        }
        if ($segmentationtableline -match $from) {
            $this.sample.info("Setting SegmentationTable setting")
            $newtableline = $segmentationtableline.replace($from, $to)
            $procedure = $procedure.replace($segmentationtableline, $newtableline)
            $procedure | Set-Content $this.algpath
        }
        return $procedure
    }
    
    <# -----------------------------------------
     CheckCoordinateSpaceIndexOption
     Checks the protocol file for the 
     coordinatespaceindex option and sets it 
     to use pixels. Since there are multiple
     lines, uses first match to fix the rest
     ------------------------------------------
     Usage: $this.CheckCoordinateSpaceIndexOption()
    ----------------------------------------- #>
    [array]CheckCoordinateSpaceIndexOption($procedure){
        $coordinatespaceline = $procedure | 
            Where-Object {$_ -match '<CoordinateSpaceIndex>'}
        if (!$coordinatespaceline) {
            throw 'error in reading <CoordinateSpaceIndex> line in procedure'
        }
        if ($coordinatespaceline -match '0') {
            $this.sample.info("Setting CoordinateSpaceIndex setting to use pixels")
            $newcoordinateline = $coordinatespaceline[0].replace('0', '1')
            $procedure = $procedure.replace($coordinatespaceline[0], $newcoordinateline)
            $procedure | Set-Content $this.algpath
        }
        return $procedure
    }
    <# -----------------------------------------
     CheckExportLineOption
     Using information from the mergeconfig csv
     file, check to make sure the outputted 
     inform prototype has the correct export 
     options and update if neccesary
     ------------------------------------------
     Usage: $this.CheckExportLineOption()
    ----------------------------------------- #>
    [void]CheckExportLineOption($procedure){
        $exportline = $procedure | 
            Where-Object {$_ -match '<exporttypes>'}
        if (!$exportline) {
            throw 'error in reading <exporttypes> line in procedure'
        }
        #
        $changedline = ''
        switch ($true) {
            {($this.needsbinaryseg -and $this.needscomponent)} {
                $changedline = $this.export_type_setting.BinaryWComponent
                break
            }
            $this.needsbinaryseg {
                $changedline = $this.export_type_setting.BinaryMaps
            }
            $this.needscomponent {
                $changedline = $this.export_type_setting.Component
            }
            default {
                $changedline = $this.export_type_setting.Default
            }
        }
        if ($exportline -ne $changedline) {
            $this.sample.info("Replacing exportline:" + $exportline + "with changedline:" + $changedline)
            $procedure.replace($exportline, $changedline) | 
                Set-Content $this.algpath
        }
    }
    <# -----------------------------------------
     GetMergeConfigData
     Get merge configuration data from 
     mergeconfig.csv and return if it already 
     exists
     ------------------------------------------
     Usage: $this.GetMergeConfigData()
    ----------------------------------------- #>
    [void]GetMergeConfigData(){
        #
        if ($this.abx -match 'Component') {
            $this.needscomponent = $true
            return
        }
        if ($this.sample.mergeconfig_data) {
            return
        }
        #
        $this.sample.ImportMergeConfigCSV($this.sample.basepath)
        $this.sample.findsegmentationtargets()
        if (!$this.sample.mergeconfig_data) {
            throw 'segmentation option not needed on any procedure'
        }
        $this.needsbinaryseg = $this.sample.binarysegtargets | 
                        Where-Object {$_.Target -contains $this.abx}
        $this.needscomponent = $this.sample.componenttarget | 
                        Where-Object {$_.Target -contains $this.abx}
        #
    }
    <# -----------------------------------------
     StartInForm
     Start an InForm process
     ------------------------------------------
     Usage: $this.StartInForm()
    ----------------------------------------- #>
    [void]StartInForm(){
        #
        $processoutputlog =  $this.outpath + "\processoutput.log"
        $arginput = " -a",  $this.algpath, `
                    "-o",  $this.informoutpath, `
                    "-i", $this.image_list_file -join ' '
        $this.sample.info("Start inForm Batch Process")
        Start-Process $this.informpath `
                -NoNewWindow `
                -RedirectStandardError $this.informprocesserrorlog `
                -PassThru `
                -ArgumentList $arginput `
                *>> $processoutputlog | Out-Null
        #
        $this.CheckForExpiredLicense()
        $this.CloseTempInformWindow()
        #
        foreach ($count in 1..20) {
            $process = get-process -name inform -EA SilentlyContinue
            if ($process) {
                break
            }
            Start-Sleep 1
        }
        #
    }
    <# -----------------------------------------
     CheckForExpiredLicense
     Check if temporary license has expired
     ------------------------------------------
     Usage: $this.CheckForExpiredLicense()
    ----------------------------------------- #>
    [void]CheckForExpiredLicense(){
        foreach ($count in 1..20) {
            $process = get-process | where-object {$_.MainWindowTitle -eq 'License Expired'}
                if ($process) {
                    $process | ForEach-Object{$_.CloseMainWindow()} | out-null
                    Start-Sleep 20
                    $this.KillinFormProcess()
                    $this.silentcleanup()
                    throw 'inForm temporary license is expired, please follow protocol to renew license'
                }
            Start-Sleep 1
        }
    }
    <# -----------------------------------------
     CloseTempInformWindow
     Let inform open and close temporary license window
     ------------------------------------------
     Usage: $this.CloseTempInformWindow()
    ----------------------------------------- #>
    [void]CloseTempInformWindow(){
        foreach ($count in 1..20) {
            $process = get-process | where-object {$_.MainWindowTitle -eq 'License Information'}
                if ($process) {
                    $process | ForEach-Object{$_.CloseMainWindow()} | out-null
                    break
                }
            Start-Sleep 1
        }
    }
    <# -----------------------------------------
     getinformtask
     Get string version of inform task
     ------------------------------------------
     Usage: $this.getinformtask()
    ----------------------------------------- #>
    [string]getinformtask() {
        $processoutputlog =  $this.outpath + "\processoutput.log"
        $arginput = " -a",  $this.algpath, `
                    "-o",  $this.informoutpath, `
                    "-i", $this.image_list_file -join ' '
        $task = $this.informpath,
                '-NoNewWindow',
                '-RedirectStandardError', $this.informprocesserrorlog,
                '-PassThru',
                '-ArgumentList',  $arginput,
                '*>>', $processoutputlog -join ' '
        return $task
    }
    <# -----------------------------------------
     WatchBatchInForm
     Check that inForm starts properly, close the 
     temporary inForm window and wait for the 
     process to complete. To make sure inForm 
     has not had any internal errors check that the
     batch.log has been written to every ten minutes
     within that same time limit. If it has not
     there is likely an error and inForm should
     be terminated.
     ------------------------------------------
     Usage: $this.WatchBatchInForm()
    ----------------------------------------- #>
    [void]WatchBatchInForm(){
        #
        if (!(test-path $this.informbatchlog) -or 
            !(get-process -name inform -EA SilentlyContinue)){
                $this.sample.warning('inForm did not properly start')
                return
            }
        $value = $true
        #
        # wait for inform to complete, if the process has 
        # not completed check the batch file
        # has been updated within that time limit. 
        # Kill inForm otherwise.
        #
        while($value){
            #
            if ((Get-ChildItem $this.informbatchlog).LastWriteTime -lt (Get-Date).AddMinutes(-10)){
                $this.sample.warning('Timeout reached for batch run')
                if (get-process -name inform -EA SilentlyContinue){
                    Throw 'Could not close failed inForm'
                } else {
                    $value = $false
                }
            } else {
                get-process -name inform -EA SilentlyContinue |
                    Wait-Process -Timeout 300 -EA SilentlyContinue -ErrorVariable value
            }
            #
        }
        $this.KillinFormProcess()
        #
    }
   <# -----------------------------------------
    CheckErrors
    Check the resulting process files for any 
    potential errors
    ------------------------------------------
    Usage: $this.CheckErrors()
   ----------------------------------------- #>
    [void]CheckErrors(){
        #
        $errs = $this.sample.GetContent($this.informprocesserrorlog)
        if ($errs){
            $this.sample.warning($errs)
        }
        #
        $batchlog = $this.sample.GetContent($this.informbatchlog)
        if (!($batchlog)){
            $this.sample.warning("inForm batch log does not exist")
            $this.err += 1
            return   
        }
        #
        if (!($batchlog -match "Batch process is completed")){
            $this.sample.warning("inForm batch log did not record a finishing event")
            $this.err += 1
            return   
        }
        #
        $this.CheckInFormOutputFiles()
        $this.CheckForKnownErrors($batchlog)
        $this.CheckForFixableFiles()
        #
    }
    <# -----------------------------------------
     CheckInFormOutputFiles
     Record the number of complete inform 
     output files of each necessary type 
     (cell_seg, binary_seg_maps, component_data)
     check if any files needed - given the 
     files found in the image list - have 
     0bytes, indicating a potential error
     ------------------------------------------
     Usage: $this.CheckInFormOutputFiles()
    ----------------------------------------- #>
    [void]CheckInFormOutputFiles(){
        #
        $informtypes = @('cell_seg_data.txt')
        if ($this.needsbinaryseg) {
            $informtypes += 'binary_seg_maps.tif'
        }
        if ($this.needscomponent) {
            $informtypes += 'component_data.tif'
        }
        #
        $this.corruptedfiles = @()
        #
        $o = $this.informoutpath+"\*"
        #
        foreach ($informtype in $informtypes) {
            #
            $ofiles = @()
            $ofiles += Get-ChildItem $o -Include ('*'+$informtype)
            $nfiles = $ofiles.Length
            if ($nfiles -ne 0) {
                $this.sample.info("inForm created " + $nfiles + " of " +
                    $this.image_list.Length + " " + $informtype + " files")
            }
            #
            $imagefiles = @()
            $imagefiles += Get-ChildItem $this.image_list -Include ('*'+$informtype) 
            foreach ($file in $imagefiles) {
                $fileneeded = $file -replace '\..*',('_'+$informtype)
                if (!$fileneeded.PSIsContainer -and $fileneeded.length -eq 0) {
                    $this.corruptedfiles += $file.FullName
                }
            }
            #
        }
        #
    }
    <# -----------------------------------------
     CheckForKnownErrors
     Check errors given from the inform batch 
     log and reference error dictionary to 
     see if image files need to be rerun or 
     skipped depending on the error
     ------------------------------------------
     Usage: $this.CheckForKnownErrors($batchlog)
    ----------------------------------------- #>
    [void]CheckForKnownErrors($batchlog){
        $completestring = 'Batch process is completed'
        $errormessage = $batchlog.Where({$_ -match $completestring}, 'SkipUntil')
        $this.sample.warning(($errormessage | Select-Object -skip 1))
        #
        $this.skippedfiles = @()
        if ($errormessage.length -gt 0) {
            foreach ($errorline in $errormessage) {
                $errorline -match '\[\d+,\d+\]'
                if ($matches) {
                    $imageid = $matches[0]
                    $this.CheckErrorDictionary($errorline, $imageid)
                }
            }
        }
    }
    <# -----------------------------------------
     CheckErrorDictionary
     check an error line against the error
     dictionary. errors can lead to files
     being rerun or skipped
     ------------------------------------------
     Usage: $this.CheckInFormOutputFiles($errorline, $imageid)
    ----------------------------------------- #>
    [void]CheckErrorDictionary($errorline, $imageid){
        #
        $filepath = $this.inputimagepath + '\' + 
                    $this.sample.slideid + '_' + $imageid + '.im3'
        switch -regex ($errorline) {
            $this.error_dictionary.ConnectionFailed {
                $this.corruptedfiles += $filepath
            }
            $this.error_dictionary.NoElements {
                $this.skippedfiles += $filepath
            }
            $this.error_dictionary.SegmentCells {
                $this.skippedfiles += $filepath
            }
            $this.error_dictionary.CorruptIM3 {
                $this.skippedfiles += $filepath
            }
            $this.error_dictionary.OverlappedObjects {
                $this.skippedfiles += $filepath
            }
        }
        #
    }
    <# -----------------------------------------
     CheckForFixableFiles
     Get corrupted files that can be fixed
     by a rerun while ommitting files that 
     should be skipped and set to new image
     list
     ------------------------------------------
     Usage: $this.CheckForFixableFiles()
    ----------------------------------------- #>
    [void]CheckForFixableFiles(){
        #
        $corrupted = $this.corruptedfiles | Select-Object -Unique
        $skipped = $this.skippedfiles | Select-Object -Unique
        $this.inputimageids = $corrupted | Where-Object {$skipped -notcontains $_}
        #
        if ($this.inputimageids.length -gt 0) {
            #
            if (!$this.islocal){
                $flatwfolder = $this.inputimagepath
                $this.sample.createnewdirs($flatwfolder)
                $this.inputimageids | ForEach-Object {
                    $source = $this.sample.flatwim3folder() + '\' + (Split-Path $_ -Leaf)
                    $this.sample.copy($source, (Split-Path $_))
                }
                #
                $this.inputimageids = @()
                #
            }
            $this.err++
        }
        else {
            $this.err = -1
        }
        #
    }
    <# -----------------------------------------
     MergeOutputDirectories
     Merge output directories to a folder using
     newest files
     ------------------------------------------
     Usage: $this.MergeOutputDirectories()
    ----------------------------------------- #>
    [void]MergeOutputDirectories(){
        $finalpath = $this.outpath + "\" + $this.abx + '_0'
        foreach($count in (1..5)) {
            $errpath = $this.outpath + "\" + $this.abx + '_' + $count
            if (Test-Path $errpath) {
                $batchfile = $errpath + '\batch.log'
                $newbatchfile = 'batch_' + $count + '.log'
                Rename-Item $batchfile $newbatchfile -ErrorAction Stop
                $errorfiles = $this.sample.listfiles($errpath, '*')
                foreach ($file in $errorfiles) {
                    $this.sample.copy($file, $finalpath)
                }
                #robocopy $errpath $finalpath -r:3 -w:3 -np -E -mt:4 /IS | out-null
            }
        }
    }
    <# -----------------------------------------
     ReturnData
     Return the data to the relevant <spath> 
     inform data location 
     ------------------------------------------
     Usage: $this.ReturnData()
    ----------------------------------------- #>
    [void]ReturnData(){
        #
        $this.sample.info("Launch data transfer")
        $this.KillinFormProcess()
        #
        $sor = $this.informoutpath
        #
        # remove legend file
        #
        $this.sample.removefile($sor, "*legend.txt")
        #
        # remove batch_procedure project and add the algorithm ##############validate##################
        #
        $this.sample.removefile($sor, '*.ifr')
        $this.sample.copy($this.algpath, $sor)
        #
        $old_name = $sor + '\' + $this.alg
        $new_name = $sor + '\' + 'batch_procedure' + 
            $this.alg.Substring($this.alg.Length-4, 4)
        Rename-Item -LiteralPath $old_name $new_name -Force
        #
        $this.sample.removedir($this.abpath)
        #
        $logfile = $this.outpath+'\robolog.log'
        $filespec = @('maps.tif', '.txt', '.ifr', '.ifp', '.log')
        $this.sample.copy($sor, $this.abpath, $filespec, 50, $logfile)
        #
        $componentimages = $this.sample.listfiles($sor, 'data.tif')
        if ($componentimages){
            $cc = $this.sample.componentfolder()
            $this.sample.removedir($cc)
            $filespec = @('data.tif', '.ifr', '.ifp', '.log')
            $this.sample.copy($sor, $cc, $filespec, 1, $logfile)
        }
        #
        $this.sample.removedir($this.outpath)
        #
        $this.sample.info("Data transfer finished")
        #
    }
    <# -----------------------------------------
     silentcleanup
     silentcleanup
     ------------------------------------------
     Usage: $this.silentcleanup()
    ----------------------------------------- #>
    [void]silentcleanup(){
        $this.sample.CreateNewDirs($this.outpath)
    }
    #
}
#