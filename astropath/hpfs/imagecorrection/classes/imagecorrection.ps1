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
Class imagecorrection : moduletools {
    #
    imagecorrection([array]$task,[launchmodule]$sample) : base ([array]$task, [launchmodule]$sample){
        $this.flevel = [FileDownloads]::BATCHID + [FileDownloads]::IM3 + [FileDownloads]::FLATFIELD + [FileDownloads]::XML
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'  
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
        $this.DownloadFiles()
        $this.ShredDat()
        $this.ApplyCorr()
        $this.InjectDat()
        $this.ExtractLayer(1)
        $this.cleanup()
    }
    <# -----------------------------------------
     TestPaths
     Test that the batch flatfield and im3 
     folder exists in the correct locations
     ------------------------------------------
     Usage: $this.TestPaths()
    ----------------------------------------- #>
    [void]TestPaths(){
        $s = $this.sample.basepath+' '+$this.sample.flatwfolder()+' '+$this.sample.slideid
        $this.sample.info($s)
        $this.sample.testim3folder()
        $this.sample.testbatchflatfield()
    }
   <# -----------------------------------------
     ApplyCor
        apply the correction
     ------------------------------------------
     Usage: $this.ApplyCor()
    ----------------------------------------- #>
    [void]ApplyCorr(){
        $this.sample.info("started applying correction")
        $taskname = 'applycorr'
        $matlabtask = ";runFlatw('"+$this.processvars[0]+"', '"+$this.processvars[1]+"', '"+$this.sample.slideid+"');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
        $this.sample.info("finished applying correction")
    }
   <# -----------------------------------------
     ExtractLayer
        Extract the particular layer of interest
     ------------------------------------------
     Usage: $this.ExtractLayer($layer)
    ----------------------------------------- #>
    [void]ExtractLayer([int]$layer){
        $this.sample.info("Extract Layer started")
        $taskname = 'extractlayer'
        $matlabtask = ";extractLayer('"+$this.processvars[1]+"', '"+$this.sample.slideid+"', '"+$layer+"');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
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
        #
        if ($this.processvars[4]){
            # im3s
            $sor = $this.processvars[0] +'\'+$this.sample.slideid+'\im3\flatw'
            $des = $this.sample.flatwim3folder()
            $this.sample.copy($sor, $des, '.im3', 30)
            if(!(((gci ($sor+'\*') -Include '*im3').Count) -eq ((gci ($des+'\*') -Include '*.im3').Count))){
                Throw 'im3s did not upload correctly'
            }
            $this.sample.copy($sor, $des, '.log')
            # fw files
            $sor = $this.processvars[1] +'\'+$this.sample.slideid
            $des = $this.sample.flatwfolder()
            $this.sample.copy($sor, $des, '.fw', 30)
            if(!(((gci ($sor+'\*') -Include '*.fw').Count) -eq ((gci ($des+'\*') -Include '*.fw').Count))){
                Throw 'fws did not upload correctly'
            }
            # fw01 files
            $this.sample.copy($sor, $des, '.fw01', 30)
            if(!(((gci ($sor+'\*') -Include '*.fw01').Count) -eq ((gci ($des+'\*') -Include '*.fw01').Count))){
                Throw 'fws did not upload correctly'
            }
            $this.sample.copy($sor, $des, '.log')
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