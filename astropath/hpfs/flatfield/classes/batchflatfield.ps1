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
        #
        #$this.sample.info('dpath: ' + $this.sample.basepath)
        #$this.sample.info('batch flatfield path: ' + $this.sample.batchflatfield())
        #$this.sample.info('slide list: ' + $slidelist)
        #
        if ($this.vers -eq '0.0.1'){
            $this.RunBatchFlatfieldMatlab()
        }
        elseif($this.vers -gt '0.0.1'){
            $this.RunBatchFlatfieldPy()
        }
    }
    <# -----------------------------------------
     RunBatchFlatfieldMatlab
        Run batch flatfield matlab code
     ------------------------------------------
     Usage: $this.RunBatchFlatfieldMatlab()
    ----------------------------------------- #>
    [void]RunBatchFlatfieldMatlab(){
        $slidelist = $this.sample.batchslides.slideID -Join ','
        $taskname = 'fltOneBatch'
        $matlabinput = "'"+$this.sample.basepath+"', '"+$this.sample.batchflatfield()+"', '"+$slidelist+"'"
        $matlabtask = ";fltOneBatch("+$matlabinput+");exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
        if (Test-Path $this.processloc) {
            Remove-Item $this.processloc -Force -Recurse -ea Stop
        }
    }
    <# -----------------------------------------
     RunBatchFlatfieldPy
        Run batch flatfield python code
     ------------------------------------------
     Usage: $this.RunBatchFlatfieldPy()
    ----------------------------------------- #>
    [void]RunBatchFlatfieldPy(){
        $taskname = 'fltOneBatch'
        $dpath = '\\bki04\Clinical_Specimen '
        #batchflatfieldcohort <Dpath>\<Dname> --sampleregex [sample_regex] --batchID [batch_ID]
        $pythontask = 'batchflatfieldcohort ' + $dpath + '--batchID ' + $this.sample.BatchID + ' --workingdir ' + $this.processloc + '\batchflatfield'
        $this.runpythontask($taskname, $pythontask)
        #if (Test-Path $this.processloc) {
        #    Remove-Item $this.processloc -Force -Recurse -ea Stop
        #}
    }
}