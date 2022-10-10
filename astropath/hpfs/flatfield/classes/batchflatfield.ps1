<#
--------------------------------------------------------
batchflatfield
Benjamin Green, Andrew Jorquera
Last Edit: 09/23/2021
--------------------------------------------------------
#>
class batchflatfield : moduletools {
    #
    batchflatfield([hashtable]$task,[launchmodule]$sample) : base ([hashtable]$task,[launchmodule]$sample){
        $this.processloc = $this.sample.basepath + '\flatfield\' + $this.sample.batchID
        $this.sample.createdirs($this.processloc)
        $this.funclocation = '"' + $PSScriptRoot + '\..\funcs"' 
    }
    <# -----------------------------------------
     RunBatchFlatfield
     Run batch flat field
     ------------------------------------------
     Usage: $this.RunBatchFlatfield()
    ----------------------------------------- #>
    [void]RunBatchFlatfield(){
        $this.GetBatchFlatfield()
        $this.datavalidation()
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
        $matlabinput = "'" + $this.sample.basepath + "', '" + 
            $this.sample.batchflatfield() + "', '" + $slidelist + "'"
        $matlabtask = ";fltOneBatch(" + $matlabinput + ");exit(0);"
        $this.runmatlabtask($taskname, $matlabtask)
        $this.silentcleanup()
    }
    <# -----------------------------------------
     silentcleanup
        silentcleanup
     ------------------------------------------
     Usage: $this.silentcleanup()
    ----------------------------------------- #>
    [void]silentcleanup(){
        #
        $this.sample.removedir($this.processloc)
        #
    }
    <# -----------------------------------------
     datavalidation
     Validation of output data
     ------------------------------------------
     Usage: $this.datavalidation()
    ----------------------------------------- #>
    [void]datavalidation(){
        if (!$this.sample.testbatchflatfield()){
            throw 'Output files are not correct'
        }
    }
}