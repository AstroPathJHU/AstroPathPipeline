﻿<#
--------------------------------------------------------
segmaps
Created By: Andrew Jorquera
Last Edit: 09/29/2021
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
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'
        $this.processloc = $this.sample.componentfolder()  
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
        $this.datavalidation()
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
            #$this.sample.removefile($sor, 'w_seg.tif')
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
        $this.sample.info("started processing segmentation maps")
        $taskname = 'GetaSeg'
        $matlabtask = ";GetaSeg('",
            $this.sample.basepath,
            "', '", $this.sample.slideid,
            "', '", $this.sample.mergeconfigfile(),
            "');exit(0);" -join ''
        $this.runmatlabtask($taskname, $matlabtask)
        $this.sample.info("finished processing segmentation maps")
    }
    <# -----------------------------------------
     GetnoSeg
        Get the component data
     ------------------------------------------
     Usage: $this.GetnoSeg()
    ----------------------------------------- #>
    [void]GetnoSeg(){
        $this.sample.info("started processing fields without segmentation data")
        $taskname = 'GetnoSeg'
        $matlabtask = ";GetnoSeg('", 
            $this.sample.basepath,
            "', '", $this.sample.slideid, 
            "', '", $this.sample.mergeconfigfile(),
            "');exit(0);" -join ''
        $this.runmatlabtask($taskname, $matlabtask)
        $this.sample.info("finished processing fields without segmentation data")
    }
    <# -----------------------------------------
     datavalidation
     Validation of output data
     ------------------------------------------
     Usage: $this.datavalidation()
    ----------------------------------------- #>
    [void]datavalidation(){
        if (!$this.sample.testsegmentationfiles()){
            throw 'Output files are not correct'
        }
    }
}
