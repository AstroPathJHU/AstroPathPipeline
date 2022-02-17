<#
--------------------------------------------------------
batchmicomp
Benjamin Green, Andrew Jorquera
Last Edit: 02.16.2022
--------------------------------------------------------
#>
class batchmicomp : moduletools {
    #
    [string]$project
    #
    batchmicomp([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'
        $this.processloc = $this.sample.basepath + '\flatfield\' + $this.sample.batchID
        $this.sample.createdirs($this.processloc)
    }
    <# -----------------------------------------
     RunBatchMeanImageComparison
     Run batch mean image comparison
     ------------------------------------------
     Usage: $this.RunBatchMeanImageComparison()
    ----------------------------------------- #>
    [void]Runbatchmicomp(){
        $this.Getbatchmicomp()
    }
    <# -----------------------------------------
     GetBatchMeanImageComparison
        Get Batch Mean Image Comparison
     ------------------------------------------
     Usage: $this.GetBatchMeanImageComparison()
    ----------------------------------------- #>
    [void]Getbatchmicomp(){
        $taskname = 'batchmicomp'
        $dpath = $this.sample.basepath
        $batchslides = $this.sample.batchslides.slideid -join '|'
        $pythontask = 'meanimagecomparison.exe ' + $dpath + " --sampleregex '" + $batchslides + "'" #+ "' --workingdir " + $this.processvars[0]
        $this.runpythontask($taskname, $pythontask)
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
}