﻿<#
--------------------------------------------------------
informinput
Created By: Benjamin Green -JHU
Last Edit: 07/23/2021
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
Usage: $a = [informinput]::new($task, $sample)
       $a.RunBatchInForm()
--------------------------------------------------------
#>
Class informinput {
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
    #
    informinput([hashtable]$task,[launchmodule]$sample) {
        #
        $this.sample = $sample
        $this.abx = $task.antibody.trim()
        $this.alg = $task.algorithm.trim()
        $this.abpath = $this.sample.phenotypefolder() + '\' + $this.abx
        $this.algpath = $this.sample.basepath +
             '\tmp_inform_data\Project_Development\' + $this.alg
        $this.informoutpath = $this.outpath + "\" + $this.abx
        $this.informpath = '"'+"C:\Program Files\Akoya\inForm\" + 
            $task.informvers.trim() + "\inForm.exe"+'"'
        $this.informbatchlog = $this.informoutpath + "\Batch.log"
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
        $this.DownloadIm3()
        while(($this.err -le 5) -AND ($this.err -ge 0)){
            #
            $this.CreateOutputDir()
            $this.CreateImageList()
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
     DownloadIm3
     Download the im3s to process to reduce network
     strain and frequent network errors while 
     processing
     ------------------------------------------
     Usage: $this.DownloadIm3()
    ----------------------------------------- #>
    [void]DownloadIm3(){
        #
        $this.sample.info("Download im3s")
        $this.sample.createnewdirs($this.outpath)
        #
        $des = $this.outpath +'\'+$this.sample.slideid+'\im3\flatw'
        $sor = $this.sample.flatwim3folder()
        #
        $this.sample.copy($sor, $des, 'im3', 30)
        if(!(((gci ($sor+'\*') -Include '*im3').Count) -eq (gci $des).count)){
            Throw 'im3s did not download correctly'
        }
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
        $this.image_list = gci -Path $p -include *.im3 |
             % {$_.FullName} |
            foreach-object {$_+"`r`n"}
        $this.sample.setfile($this.image_list_file, $this.image_list)
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
        $prc = Start-Process $this.informpath `
                -NoNewWindow `
                -RedirectStandardError $this.informprocesserrorlog `
                -PassThru `
                -ArgumentList $arginput `
                *>> $processoutputlog
        #
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
        # let inform open and close temporary license window
        #
        Start-Sleep 20
        get-process |
            where-object {$_.MainWindowTitle -eq 'License Information'} |
            %{$_.CloseMainWindow()} | out-null
        Start-Sleep 20
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
            if ((gci $this.informbatchlog).LastWriteTime -lt (Get-Date).AddMinutes(-10)){
                $this.KillinFormProcess()
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
            $this.err = $this.err + 1
            return   
        }
        #
        if (!($batch -match "Batch process is completed")){
            $this.sample.warning("inForm batch log did not record a finishing event")
            $this.err = $this.err + 1
            return   
        }
        #
        $this.CheckInFormOutputFiles()
        #
    }
    <# -----------------------------------------
     CheckInFormOutputFiles
     record the number of complete inform 
     output files of each type 
     (cell_seg, binary_seg_maps, component_data)
     and check if any files have 0bytes, 
     indicating a potential error.
     ------------------------------------------
     Usage: $this.CheckInFormOutputFiles()
    ----------------------------------------- #>
    [void]CheckInFormOutputFiles(){
        #
        $o = $this.informoutpath+"\*"
        $informtypes = @('cell_seg_data.txt','binary_seg_maps.tif','component_data.tif')
        #
        foreach($informtype in $informtypes){
            #
            $informtype = '*'+$informtype
            $ofiles = gci $o -Include ('*'+$informtype)
            $nfiles = $ofiles.Length
            if ($nfiles -ne 0){
                $this.sample.info("inForm created "+$nfiles+" of "+`
                    $this.image_list.Length+" "+$informtype+" files")
                #
                $b = ($ofiles| Measure Length -Minimum).Minimum
                if ($b -eq 0kb){
                    $this.sample.warning("Some "+$informtype+" files appear to be corrupt")
                    $this.err = $this.err + 1
                    return
                }
            }
            #
        }
        #
        $this.err = -1
        #
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