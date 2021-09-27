<#
--------------------------------------------------------
batchflatfield
Created By: Andrew Jorquera
Last Edit: 09/23/2021
--------------------------------------------------------
#>
class batchflatfield : moduletools {
    #
    [Array]$cleanedtasks
    [string]$project
    #
    batchflatfield([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'  
    }
    <# -----------------------------------------
     RunBatchFlatField
     Run batch flat field
     ------------------------------------------
     Usage: $this.RunBatchFlatField()
    ----------------------------------------- #>
    [void]RunBatchFlatField(){
        $this.slidelookup()
        $this.GetBatchFlatField()
        $this.returndata()
    }
    <# -----------------------------------------
     slidelookup
     Run slide lookup
     ------------------------------------------
     Usage: $this.slidelookup()
    ----------------------------------------- #>
    [void]slidelookup(){
        #
        $slides = $this.importslideids($this.mpath)
        $project_dat = $this.ImportConfigInfo($this.mpath)
        #
        # select samples from the appropriate modules 
        #
        if ($this.project -eq $null){
            $projects = ($project_dat | Where-object {$_.($this.module) -match 'yes'}).Project
        } else {
            $projects = $this.project
        }
        #
        $cleanedslides = $slides | Where-Object {$projects -contains $_.Project -and $batchID -contains $_.BatchID}
    }
    <# -----------------------------------------
     GetBatchFlatField
        Get Batch flat field
     ------------------------------------------
     Usage: $this.GetBatchFlatField()
    ----------------------------------------- #>
    [void]GetBatchFlatField(){
        $this.sample.info("started flatfield batch")
        $taskname = 'fltOneBatch'
        $matlabtask = ";fltOneBatch('"+$this.sample.basepath+"', '"+$this.sample.batchflatfield()+"');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
        $this.sample.info("finished flatfield batch")
    }
    <# -----------------------------------------
     returndata
     returns data to flatfield folder
     ------------------------------------------
     Usage: $this.returndata()
    ----------------------------------------- #>
    [void]returndata(){
        #
        $this.sample.info("Return data started")

        $sor = $this.processvars[1]
		$des = $this.sample.basepath + '\flatfield'
        
        xcopy $sor, $des /q /y /z /j /v | Out-Null
        
        $this.sample.info("Return data finished")
        #
    }
}