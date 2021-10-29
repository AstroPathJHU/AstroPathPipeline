﻿<#
--------------------------------------------------------
shredxml
Created By: Andrew Jorquera
Last Edit: 09/20/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[array]: the 3 part array of project, slideid, process loc
    E.g. @('7','M18_1','\\bki08\e$')
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [shredxml]::new($task, $sample)
       $a.runshredxml()
--------------------------------------------------------
#>
Class shredxml : moduletools {
    #
    shredxml([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.processvars[0] = $this.sample.basepath
        $this.sample.CreateDirs($this.processloc)
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'  
    }
    <# -----------------------------------------
     RunShredXML
     Run shred xml
     ------------------------------------------
     Usage: $this.RunShredXML()
    ----------------------------------------- #>
    [void]RunShredXML(){
        $this.ShredXML()
        $this.returndata()
        $this.cleanup()
    }
    <# -----------------------------------------
     returndata
     returns data to source path
     ------------------------------------------
     Usage: $this.returndata()
    ----------------------------------------- #>
    [void]returndata(){
        #
        $this.sample.info("Return data started")
        #
		$sor = $this.processvars[1] + '\' + $this.sample.slideid
		$des = $this.sample.xmlfolder()
		$this.sample.copy($sor, $des, 'xml', 30)
        $this.sample.copy($sor, $des, 'log')
        $this.sample.info("Return data finished")
        #
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
            gci $this.processloc -Recurse | Remove-Item -force -recurse
            remove-item $this.processloc -force
        } else {
            gci ($this.processloc+'\*') -Include '*.xml' -Recurse | Remove-Item -force -recurse
        }
        #
        $this.sample.info("cleanup finished")
        #
    }
}
