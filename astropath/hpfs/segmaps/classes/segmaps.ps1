<#
--------------------------------------------------------
segmaps
Created By: Andrew Jorquera
Last Edit: 09/21/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[array]: the 3 part array of project, slideid, process loc
    E.g. @('7','M18_1','\\bki08\e$')
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [segmaps]::new($task, $sample)
       $a.runsegmaps()
--------------------------------------------------------
#>
Class segmaps : moduletools {
    #
    segmaps([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.flevel = [FileDownloads]::IM3
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'  
    }
    <# -----------------------------------------
     RunSegMaps
     Run seg maps
     ------------------------------------------
     Usage: $this.RunSegMaps()
    ----------------------------------------- #>
    [void]RunSegMaps(){
        $this.cleanup()
        $this.GetaSeg()
        $this.GetnoSeg()
    }
    <# -----------------------------------------
     cleanup
     cleanup the data directory
     ------------------------------------------
     Usage: $this.cleanup()
    ----------------------------------------- #>
    [void]cleanup(){
        #
        $this.sample.info("cleanup started")
        #
        if ($this.processvars[4]){
            $sor = $this.sample.componentfolder()
            Get-ChildItem -Path $sor -Include *w_seg.tif -Recurse | Remove-Item -force
        }
        $this.sample.info("cleanup finished")
        #
    }
   <# -----------------------------------------
     GetaSeg
        Get the seg maps
     ------------------------------------------
     Usage: $this.GetaSeg()
    ----------------------------------------- #>
    [void]GetaSeg(){
        $this.sample.info("started getting seg maps")
        $taskname = 'GetaSeg'
        $matlabtask = ";GetaSeg('"+$this.processvars[0]+"', '"+$this.sample.slideid+"', '"+$this.sample.mergeconfig+"');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
        $this.sample.info("finished getting seg maps")
    }
   <# -----------------------------------------
     GetnoSeg
        Get the component data
     ------------------------------------------
     Usage: $this.GetnoSeg()
    ----------------------------------------- #>
    [void]GetnoSeg(){
        $this.sample.info("started getting component data")
        $taskname = 'GetnoSeg'
        $matlabtask = ";GetnoSeg('"+$this.processvars[0]+"', '"+$this.sample.slideid+"', '"+$this.sample.mergeconfig+"');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
        $this.sample.info("finished getting component data")
    }
}
