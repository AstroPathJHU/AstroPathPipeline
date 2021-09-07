<#
--------------------------------------------------------
informinput
Created By: Benjamin Green -JHU
Last Edit: 07/23/2021
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
Class informinput {
    #
    [string]$stringin
    [string]$abx
    [string]$alg
    [string]$abpath
    [string]$algpath
    [string]$outpath
    [string]$informoutpath
    [string]$image_list_file
    [array]$image_list
    [string]$informpath
    [launchmodule]$sample
    [string]$informbatchlog
    [int]$err
    [string]$informprocesserrorlog
    [string]$vers
    #
    informinput([array]$task,[launchmodule]$sample){
        #
        $this.sample = $sample
        $this.abx = $task[2].trim()
        $this.alg = $task[3].trim()
        $this.abpath = $this.sample.phenotypefolder()+'\'+$this.abx
        $this.algpath = $this.sample.basepath+'\tmp_inform_data\Project_Development\'+$this.alg
        $this.outpath = "C:\Users\Public\BatchProcessing"
        $this.informoutpath = $this.outpath+"\"+$this.abx
        $this.informprocesserrorlog =  $this.outpath+"\informprocesserror.log"
        $this.informbatchlog = $this.informoutpath+"\Batch.log"
        $this.image_list_file = $this.outpath+"\image_list.tmp"
        $this.vers = $task[4].trim()
        $this.informpath = '"'+"C:\Program Files\Akoya\inForm\"+$this.vers+"\inForm.exe"+'"'
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
        <#
        if (!(test-path $this.informpath)){
            Throw "inform path not found for:" + $this.vers
        }
        #>
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
        #
        $this.sample.info("Create inForm output location")
        $tQ = test-path $this.informoutpath
        if ($tQ){
                remove-item $this.informoutpath -force -Recurse -EA STOP
            }
        New-Item $this.informoutpath -itemtype "directory" -EA STOP | Out-NULL
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
        $value = get-process -name inform -EA SilentlyContinue |
                                    Stop-Process -Force -EA stop
        $value2 = get-process -name rserve -EA SilentlyContinue |
                    Stop-Process -Force -EA stop
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
        $tQ = test-path $this.outpath
        if ($tQ){
                remove-item $this.outpath -force -Recurse -EA STOP
            }
        New-Item $this.outpath -itemtype "directory" -EA STOP | Out-NULL
        #
        $des = $this.outpath +'\'+$this.sample.slideid+'\im3\flatw'
        $sor = $this.sample.flatwim3folder()
        robocopy $sor $des *im3 -r:3 -w:3 -np -mt:30 |out-null
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
        $p = $this.outpath +'\'+$this.sample.slideid+'\im3\flatw\*'
        $this.image_list = gci -Path $p -include *.im3 | % {$_.FullName}
        Set-Content $this.image_list_file $this.image_list -EA STOP
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
        $processoutputlog =  $this.outpath+"\processoutput.log"
        $arginput = " -a "+$this.algpath+" -o "+$this.informoutpath+" -i "+$this.image_list_file
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
        if (!(test-path $this.informbatchlog) `
            -or !(get-process -name inform -EA SilentlyContinue)){
                Throw 'inForm did not properly start'
            }
        $value = $true
        #
        # wait for inform to complete, if the process has not completed check the batch file
        # has been updated within that time limit. Kill inForm otherwise.
        #
        while($value){
                #
                if ((gci $this.informbatchlog).LastWriteTime -lt (Get-Date).AddMinutes(-10)){
                    $this.KillinFormProcess()
                    $this.sample.warning('Timeout reached for batch run')
                    if (get-process -name inform -EA SilentlyContinue){
                        Throw 'Could not close failed inForm'
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
        $sor1 = $sor+'\*'
        #
        # remove legend file
        #
        $sor2 = gci $sor1 -include "*legend.txt" | % {$_.FullName}
        try {Remove-Item $sor2 -Force -EA SilentlyContinue} catch {}
        #
        # remove batch_procedure project and add the algorithm ##############validate##################
        #
        $sor3 = gci $sor1 -include "*.ifr" | % {$_.FullName}
        try {Remove-Item $sor3 -Force -EA SilentlyContinue} catch {}
        XCOPY /q /y /z $this.algpath $sor 
        $old_name = $sor+'\'+$this.alg
        $new_name = $sor+'\'+'batch_procedure'+$this.alg.Substring($this.alg.Length-4, 4)
        Rename-Item -LiteralPath $old_name $new_name -Force
        #
        if (test-path $this.abpath){
            remove-item $this.abpath -force -Recurse -EA SilentlyContinue   
        }
        #
        $logfile = $this.outpath+'\robolog.log'
        $moutput = robocopy $sor $this.abpath *maps.tif *.txt *.ifr *.ifp *.log -r:3 -w:3 -np -mt:50 -log:$logfile
        #
        $sor4 = gci $sor1 -include "*data.tif"
        if ($sor4){
            $cc = $this.sample.componentfolder()
            if (test-path $cc) {
                remove-item $cc -Force -Recurse -EA SilentlyContinue
            }
            $moutput = robocopy $sor $cc *data.tif *.ifr *.ifp *.log -r:3 -w:3 -np -mt:1 -log:$logfile
        }
        #
        Remove-Item $this.outpath -Recurse -Force -EA STOP
        #
        $this.sample.info("Data transfer finished")
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
#
}
#