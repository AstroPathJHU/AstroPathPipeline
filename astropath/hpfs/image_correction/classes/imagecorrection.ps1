<#
--------------------------------------------------------
imagecorrection
Created By: Benjamin Green -JHU
Last Edit: 07/23/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[array]: the 3 part array of project, slideid, process loc
    E.g. @('7','M18_1','\\bki08\e$')
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [imagecorrection]::new($task, $sample)
       $a.runimagecorrection()
--------------------------------------------------------
#>
Class imagecorrection {
    #
    [string]$processloc
    [launchmodule]$sample
    [array]$processvars
    [string]$convertim3pathlogshred
    [string]$convertim3pathloginject
    [string]$applycorlog
    [string]$extractlayerlog
    [int]$downloadim3ii = 0
    [string]$vers
    [string]$funclocation = '"'+$PSScriptRoot + '\..\funcs"'

    #
    imagecorrection([array]$task,[launchmodule]$sample){
        #
        $this.sample = $sample
        $this.BuildProcessLocPaths($task)
        $this.vers = $this.sample.GetVersion($this.sample.mpath, $this.sample.module, $task[0])   
        #
    }
    <# -----------------------------------------
     BuildProcessLocPath
     build the processing specimens directory paths
     if it does not exist. If input is '*' then
     work 'in place'
     ------------------------------------------
     Usage: $this.BuildProcessLoc()
    ----------------------------------------- #>
    [void]BuildProcessLocPaths($task){
        #
        $fwpath = '\\'+$this.sample.project_data.fwpath
        $this.processvars = @($this.sample.basepath, $fwpath, $this.sample.flatwim3folder(), $this.sample.batchflatfield())
        #
        # If processloc is not '*' a processing destination was added as input, correct the paths to analyze from there
        #
        if ($this.processloc -ne '*'){
             $this.processloc = ($task[2]+'\processing_imagecorrection\'+$task[1])
             #
             $this.convertim3pathlogshred = $this.processloc + '\convertIM3pathshred.log'
             $this.convertim3pathloginject = $this.processloc + '\convertIM3pathinject.log'
             $this.applycorlog = $this.processloc + '\applycor.log'
             $this.extractlayerlog = $this.processloc + '\extractlayer.log'
             #
             $this.processvars = $this.processvars -replace [regex]::escape($this.sample.basepath), $this.processloc  
             $this.processvars = $this.processvars -replace [regex]::escape('\\'+$this.sample.project_data.fwpath), ($this.processloc+'\flatw')
             $this.downloadim3ii = 1
        } else {
             $this.convertim3pathlogshred = $this.sample.flatwfolder() + '\convertIM3pathshred.log'
             $this.convertim3pathloginject = $this.sample.flatwfolder() + '\convertIM3pathinject.log'
             $this.applycorlog = $this.sample.flatwfolder() + '\applycor.log'
             $this.extractlayerlog = $this.sample.flatwfolder() + '\extractlayer.log'
        }
        #
    }
    <# -----------------------------------------
     RunImageCorrection
     Run image correction
     ------------------------------------------
     Usage: $this.runimagecorrection()
    ----------------------------------------- #>
    [void]RunImageCorrection(){
        $this.TestPaths()
        $this.fixM2()
        $this.DownloadIm3()
        $this.ShredDat()
        $this.ApplyCorr()
        $this.InjectDat()
        $this.ExtractLayer(1)
        $this.CleanUp()
    }
    <# -----------------------------------------
     TestPaths
     Test that the batch flatfield and im3 
     folder exists in the correct locations
     ------------------------------------------
     Usage: $this.TestPaths()
    ----------------------------------------- #>
    [void]TestPaths(){
        #
        $s = $this.sample.basepath+' '+$this.sample.flatwfolder()+' '+$this.sample.slideid
        $this.sample.info($s)
         if (!(test-path $this.sample.im3folder())){
            Throw "im3 folder not found for:" + $this.sample.im3folder()
        }
        if (!(test-path $this.sample.batchflatfield())){
            Throw "batch flatfield not found for:" + $this.sample.batchflatfield()
        }
    }
    <# -----------------------------------------
     DownloadIm3
     Download the im3s to process; reduces network
     strain and frequent network errors while 
     processing
     ------------------------------------------
     Usage: $this.DownloadIm3()
    ----------------------------------------- #>
    [void]DownloadIm3(){
        #
        if ($this.downloadim3ii -eq 1){
            $this.sample.info("Download im3s started")
            foreach($ii in @(0,1,2)){
                if (test-path $this.processvars[$ii]){
                        remove-item $this.processvars[$ii] -force -Recurse -EA STOP
                    }
                New-Item $this.processvars[$ii] -itemtype "directory" -EA STOP | Out-NULL
            }
            # flatfield
            $flatfieldfolder = $this.processvars[0]+'\flatfield'
            if (test-path $flatfieldfolder){
                    remove-item $flatfieldfolder -force -Recurse -EA STOP
                }
            New-Item $flatfieldfolder -itemtype "directory" -EA STOP | Out-NULL
            xcopy $this.sample.batchflatfield(), $flatfieldfolder /q /y /z /j /v | Out-Null
            # im3s
            $des = $this.processvars[0] +'\'+$this.sample.slideid+'\im3\'+$this.sample.Scan()+,'\MSI'
            $sor = $this.sample.MSIfolder()
            robocopy $sor $des *im3 -r:3 -w:3 -np -mt:30 |out-null
            if(!(((gci ($sor+'\*') -Include '*im3').Count) -eq (gci $des).count)){
                Throw 'im3s did not download correctly'
            }
            # batchid
            $des = $this.processvars[0] +'\'+$this.sample.slideid+'\im3\'+$this.sample.Scan()
            xcopy $this.sample.batchIDfile(), $des /q /y /z /j /v | Out-Null
            $this.sample.info("Download im3s finished")
            #
        }
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
   <# -----------------------------------------
     ShredDat
        Extract data.dat and xml files
     ------------------------------------------
     Usage: $this.ShredDat()
    ----------------------------------------- #>
    [void]ShredDat(){
        $this.sample.info("Shred data started")
        ConvertIM3Path $this.processvars[0] $this.processvars[1] $this.sample.slideid -s -verbose 4>&1 >> $this.convertim3pathlogshred
        $log = $this.sample.GetContent($this.convertim3pathlogshred) |
             where-object {$_ -notlike '.*' -and $_ -notlike '*PM*'} | 
             foreach {$_.trim()}
        $this.sample.info($log)
        remove-item $this.convertim3pathlogshred -force -ea Continue
        $this.sample.info("Shred data finished")
    }
   <# -----------------------------------------
     ApplyCor
        apply the correction
     ------------------------------------------
     Usage: $this.ApplyCor()
    ----------------------------------------- #>
    [void]ApplyCorr(){
        $this.sample.info("started applying correction")
        $applytask = ";runFlatw('"+$this.processvars[0]+"', '"+$this.processvars[1]+"', '"+$this.sample.slideid+"');exit(0);"
        matlab -nosplash -nodesktop -minimize -sd $this.funclocation -r $applytask -wait >> $this.applycorlog
        if (test-path $this.applycorlog){
            remove-item $this.applycorlog -force -ea Continue
        }
        $this.sample.info("finished applying correction")
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
        $this.sample.info("Inject data started")
        ConvertIM3Path $this.processvars[0] $this.processvars[1] $this.sample.slideid -i -verbose 4>&1 >> $this.convertim3pathloginject
        $log = $this.sample.GetContent($this.convertim3pathloginject) |
             where-object {$_ -notlike '.*' -and $_ -notlike '*PM*'} | 
             foreach {$_.trim()}
        $this.sample.info($log)
        remove-item $this.convertim3pathloginject -force -ea Continue
        $this.sample.info("Inject data finished")
    }
   <# -----------------------------------------
     ExtractLayer
        Extract the particular layer of interest
     ------------------------------------------
     Usage: $this.ExtractLayer($layer)
    ----------------------------------------- #>
    [void]ExtractLayer([int]$layer){
        $this.sample.info("Extract Layer started")
        $applytask = ";extractLayer('"+$this.processvars[1]+"', '"+$this.sample.slideid+"', '"+$layer+"');exit(0);"
        matlab -nosplash -nodesktop -minimize -sd $this.funclocation -r $applytask -wait >> $this.extractLayerlog
        if (test-path $this.extractLayerlog){
            remove-item $this.extractLayerlog -force -ea Continue
        }
        $this.sample.info("Extract Layer finished")
    }
    <# -----------------------------------------
     cleanup
     cleanup the data directory and return the 
     data to the dpath locations
     ------------------------------------------
     Usage: $this.cleanup()
    ----------------------------------------- #>
    [void]cleanup(){
        #
        $this.sample.info("cleanup started")
        # xml files
        $sor = $this.processvars[1] +'\'+$this.sample.slideid
        $des = $this.sample.xmlfolder()
        robocopy $sor $des *.xml -r:3 -w:3 -np -mt:30 |out-null
        robocopy $sor $des *log -r:3 -w:3 -np -mt:1 |out-null
        gci ($sor+'\*') -include '*.xml' | Remove-Item -force
        #
        if ($this.downloadim3ii -eq 1){
            # im3s
            $sor = $this.processvars[0] +'\'+$this.sample.slideid+'\im3\flatw'
            $des = $this.sample.flatwim3folder()
            robocopy $sor $des *im3 -r:3 -w:3 -np -mt:30 |out-null
            if(!(((gci ($sor+'\*') -Include '*im3').Count) -eq ((gci ($des+'\*') -Include '*.im3').Count))){
                Throw 'im3s did not upload correctly'
            }
            robocopy $sor $des *log -r:3 -w:3 -np -mt:1 |out-null
            # fw files
            $sor = $this.processvars[1] +'\'+$this.sample.slideid
            $des = $this.sample.flatwfolder()
            robocopy $sor $des *.fw -r:3 -w:3 -np -mt:30 |out-null
            if(!(((gci ($sor+'\*') -Include '*.fw').Count) -eq ((gci ($des+'\*') -Include '*.fw').Count))){
                Throw 'fws did not upload correctly'
            }
            # fw01 files
            robocopy $sor $des *.fw01 -r:3 -w:3 -np -mt:30 |out-null
            if(!(((gci ($sor+'\*') -Include '*.fw01').Count) -eq ((gci ($des+'\*') -Include '*.fw01').Count))){
                Throw 'fws did not upload correctly'
            }
            robocopy $sor $des *log -r:3 -w:3 -np -mt:1 |out-null
            #
            Get-ChildItem -Path $this.processloc -Recurse | Remove-Item -force -recurse
            remove-item $this.processloc -force
            #
        }
        #
        $this.sample.info("cleanup finished")
        #
    }
}