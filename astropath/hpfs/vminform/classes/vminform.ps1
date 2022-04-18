﻿<#
--------------------------------------------------------
informinput
Created By: Benjamin Green, Andrew Jorquera
Last Edit: 03/28/2022
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[array]: the 5 part array of dpath, slideid, 
    antibody, algorithm, and inform version to use
    E.g. @('\\bki04\Clinical_Specimen_2','M18_1','CD8,CD8_12.05.2018_highTH.ifr','2.4.8.)
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [informinput]::new($task, $sample)
       $a.RunBatchInForm()
--------------------------------------------------------
#>
Class informinput : moduletools {
    #
    [string]$stringin
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
    [bool]$needsbinaryseg
    [bool]$needscomponent
    #
    $export_type_setting = @{
        Default          = '     <ExportTypes>eet_NucSegmentation, eet_CytoSegmentation, eet_MembraneSegmentation</ExportTypes>';
        BinaryMaps       = '     <ExportTypes>eet_Segmentation, eet_NucSegmentation, eet_CytoSegmentation, eet_MembraneSegmentation</ExportTypes>';
        Component        = '     <ExportTypes>eet_NucSegmentation, eet_CytoSegmentation, eet_MembraneSegmentation, eet_ComponentData</ExportTypes>';
        BinaryWComponent = '     <ExportTypes>eet_Segmentation, eet_NucSegmentation, eet_CytoSegmentation, eet_MembraneSegmentation, eet_ComponentData</ExportTypes>'
    }

    #
    informinput([array]$task, [launchmodule]$sample) : base ([array]$task, [launchmodule]$sample){
        #
        $this.flevel = [FileDownloads]::FLATWIM3
        #
        $this.sample = $sample
        $this.abx = $task[2].trim()
        $this.alg = $task[3].trim()
        $this.abpath = $this.sample.phenotypefolder() + '\' + $this.abx
        $this.algpath = $this.sample.basepath +
             '\tmp_inform_data\Project_Development\' + $this.alg
        $this.informoutpath = $this.outpath + "\" + $this.abx
        $this.informpath = '"'+"C:\Program Files\Akoya\inForm\" + 
            $task[4].trim() + "\inForm.exe"+'"'
        $this.informbatchlog = $this.informoutpath + "\Batch.log"
        $this.processvars[0] = $this.outpath
        $this.processvars[1] = $this.outpath
        $this.processvars[2] = $this.outpath
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
     RunBatchInForm
     Run the batch process 
     ------------------------------------------
     Usage: $this.RunBatchInForm()
    ----------------------------------------- #>
    [void]RunBatchInForm(){
        #
        $this.sample.createnewdirs($this.outpath)
        $this.DownloadFiles()
        while(($this.err -le 5) -AND ($this.err -ge 0)){
            #
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
            }
        }
        #
        $this.ReturnData()
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
        $this.KillinFormProcess()
        $this.sample.info("Create inForm output location")
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
                                    Stop-Process -Force -EA stop
        get-process -name rserve -EA SilentlyContinue |
                    Stop-Process -Force -EA stop
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
        $p = $this.outpath + '\' + $this.sample.slideid + '\im3\flatw\*'
        $this.image_list = Get-ChildItem -Path $p -include *.im3 |
             ForEach-Object {$_.FullName} |
            foreach-object {$_+"`r`n"}
        $this.sample.setfile($this.image_list_file, $this.image_list)
        #
    }
    <# -----------------------------------------
     CheckExportOptions
     using information from the mergeconfig csv
     file, check to make sure the outputted 
     inform prototype has the correct export 
     options and update if neccesary
     ------------------------------------------
     Usage: $this.CheckExportOptions()
    ----------------------------------------- #>
    [void]CheckExportOptions(){
        #
        $this.GetSegmentationData()
        #
        $procedure = $this.sample.GetContent($this.algpath)
        $exportline = $procedure | Where-Object {$_ -match '<exporttypes>'}
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
        $this.sample.info("Replacing exportline:" + $exportline + "with changedline:" + $changedline)
        (Get-Content $this.algpath).replace($exportline, $changedline) | 
            Set-Content $this.algpath
        #
    }
    
    <# -----------------------------------------
     GetSegmentationData
     Get segmentation data from mergeconfig.csv
     and return if it already exists
     ------------------------------------------
     Usage: $this.GetSegmentationData()
    ----------------------------------------- #>
    [void]GetSegmentationData(){
        #
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
        # let inform open and close temporary license window
        #
        foreach ($count in 1..20) {
            $process = get-process | where-object {$_.MainWindowTitle -eq 'License Information'}
                if ($process) {
                    $process | ForEach-Object{$_.CloseMainWindow()} | out-null
                    break
                }
            Start-Sleep 1
        }
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
    #
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
    check the resulting process files for any 
    potential errors
    ------------------------------------------
    Usage: $this.CheckErrors()
   ----------------------------------------- #>
    [void]CheckErrors(){
        #
        $errs = Get-Content $this.informprocesserrorlog
        if ($errs){
            $this.sample.warning($errs)
        }
        #
        $batch = Get-Content $this.informbatchlog
        if (!($batch)){
            $this.sample.warning("inForm batch log does not exist")
            $this.err += 1
            return   
        }
        #
        if (!($batch -match "Batch process is completed")){
            $this.sample.warning("inForm batch log did not record a finishing event")
            $this.err += 1
            return   
        }
        #
        $this.CheckInFormOutputFiles()
        #
    }
    <# -----------------------------------------
     CheckInFormOutputFiles
     record the number of complete inform 
     output files of each necessary type 
     (cell_seg, binary_seg_maps, component_data)
     and check if any files have 0bytes, 
     indicating a potential error.
     ------------------------------------------
     Usage: $this.CheckInFormOutputFiles()
    ----------------------------------------- #>
    [void]CheckInFormOutputFiles(){
        #
        $o = $this.informoutpath+"\*"
        $informtypes = @('cell_seg_data.txt')
        if ($this.needsbinaryseg) {
            $informtypes += 'binary_seg_maps.tif'
        }
        if ($this.needscomponent) {
            $informtypes += 'component_data.tif'
        }
        Write-Host '    InformTypes:' $informtypes
        #
        $this.corruptedfiles = @()
        #
        foreach ($informtype in $informtypes) {
            #
            $informtype = '*'+$informtype
            $ofiles = @()
            $ofiles += Get-ChildItem $o -Include ('*'+$informtype)
            $nfiles = $ofiles.Length
            if ($nfiles -ne 0) {
                $this.sample.info("inForm created " + $nfiles + " of " +
                    $this.image_list.Length + " " + $informtype + " files")
                $ofiles | foreach-object {
                    if (!$_.PSIsContainer -and $_.length -eq 0) {
                        $this.corruptedfiles += $_.FullName
                    }
                }
            }
            #
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
#
}
#