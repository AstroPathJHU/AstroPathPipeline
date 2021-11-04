<#
--------------------------------------------------------
batchmeanimagecomparison
Created By: Andrew Jorquera
Last Edit: 10/27/2021
--------------------------------------------------------
#>
class batchmeanimagecomparison : moduletools {
    #
    [string]$project
    #
    batchmeanimagecomparison([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'
        $this.processloc = $this.sample.basepath + '\flatfield\' + $this.sample.batchID
        if (!(Test-Path $this.processloc)) {
            New-Item $this.processloc -ItemType 'directory' | Out-Null
        }
    }
    <# -----------------------------------------
     RunBatchMeanImageComparison
     Run batch mean image comparison
     ------------------------------------------
     Usage: $this.RunBatchMeanImageComparison()
    ----------------------------------------- #>
    [void]RunBatchMeanImageComparison(){
        $this.GetBatchMeanImageComparison()
    }
    <# -----------------------------------------
     GetBatchMeanImageComparison
        Get Batch Mean Image Comparison
     ------------------------------------------
     Usage: $this.GetBatchMeanImageComparison()
    ----------------------------------------- #>
    [void]GetBatchMeanImageComparison(){
        $taskname = 'batchmeanimagecomparison'
        #$dpath = $this.sample.basepath
        $dpath = '\\bki04\Clinical_Specimen'
        $batchslides = $this.sample.batchslides.slideid -join '|'
        $pythontask = 'meanimagecomparison.exe ' + $dpath + " --sampleregex '" + $batchslides + "'" #+ "' --workingdir " + $this.processvars[0]
        $this.runpythontask($taskname, $pythontask)
        if (Test-Path $this.processloc) {
            Remove-Item $this.processloc -Force -Recurse -ea Stop
        }
    }
}