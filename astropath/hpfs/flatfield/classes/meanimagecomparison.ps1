<#
--------------------------------------------------------
meanimagecomparison
Created By: Andrew Jorquera
Last Edit: 10/13/2021
--------------------------------------------------------
#>
class meanimagecomparison : moduletools {
    #
    [string]$project
    #
    meanimagecomparison([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'
        $this.processloc = $this.sample.basepath + '\flatfield\' + $this.sample.batchID
        if (!(Test-Path $this.processloc)) {
            New-Item $this.processloc -ItemType 'directory' | Out-Null
        }
    }
    <# -----------------------------------------
     RunMeanImageComparison
     Run mean image comparison
     ------------------------------------------
     Usage: $this.RunMeanImageComparison()
    ----------------------------------------- #>
    [void]RunMeanImageComparison(){
        $this.GetMeanImageComparison()
    }
    <# -----------------------------------------
     GetMeanImageComparison
        Get Mean Image Comparison
     ------------------------------------------
     Usage: $this.GetMeanImageComparison()
    ----------------------------------------- #>
    [void]GetMeanImageComparison(){
        $taskname = 'meanimagecomparison'
        $dpath = '\\bki04\Clinical_Specimen '
        #meanimagecomparison --root-dirs [Dpaths] --sampleregex [sample_regex]
        $pythontask = 'meanimagecomparison --root-dirs ' + $dpath + ' --sampleregex ' + ($this.sample.batchslides.slideid -join '|')
        $this.runpythontask($taskname, $pythontask)
        #if (Test-Path $this.processloc) {
        #    Remove-Item $this.processloc -Force -Recurse -ea Stop
        #}
    }
}