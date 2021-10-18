<#
--------------------------------------------------------
batchflatfield
Created By: Andrew Jorquera
Last Edit: 09/23/2021
--------------------------------------------------------
#>
class batchflatfield : moduletools {
    #
    [string]$project
    #
    batchflatfield([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'
        $this.processloc = $this.sample.basepath + '\flatfield\' + $this.sample.batchID
        if (!(Test-Path $this.processloc)) {
            New-Item $this.processloc -ItemType 'directory' | Out-Null
        }
    }
    <# -----------------------------------------
     RunBatchFlatfield
     Run batch flat field
     ------------------------------------------
     Usage: $this.RunBatchFlatfield()
    ----------------------------------------- #>
    [void]RunBatchFlatfield(){
        $this.GetBatchFlatfield()
    }
    <# -----------------------------------------
     GetBatchFlatField
        Get Batch flat field
     ------------------------------------------
     Usage: $this.GetBatchFlatfield()
    ----------------------------------------- #>
    [void]GetBatchFlatfield(){
        $slidelist = $this.sample.batchslides.slideID -Join ','
        $taskname = 'fltOneBatch'
        $matlabinput = "'"+$this.sample.basepath+"', '"+$this.sample.batchflatfield()+"', '"+$slidelist+"'"
        $matlabtask = ";fltOneBatch("+$matlabinput+");exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
        if (Test-Path $this.processloc) {
            Remove-Item $this.processloc -Force -Recurse -ea Stop
        }
    }
}