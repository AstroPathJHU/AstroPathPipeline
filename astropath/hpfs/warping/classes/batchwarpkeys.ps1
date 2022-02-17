<#
--------------------------------------------------------
batchmeanimagecomparison
Benjamin Green, Andrew Jorquera
Last Edit: 02.16.2022
--------------------------------------------------------
#>
class batchwarpkeys : moduletools {
    #
    [string]$project
    [switch]$all = $true
    #
    batchwarpkeys([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.processloc = $this.sample.basepath + '\flatfield\' + $this.sample.batchID
        $this.sample.createdirs($this.processloc)
    }
    <# -----------------------------------------
     RunBatchMeanImageComparison
     Run batch mean image comparison
     ------------------------------------------
     Usage: $this.RunBatchMeanImageComparison()
    ----------------------------------------- #>
    [void]Runbatchwarpkeys(){
        $this.Getbatchwarpkeys()
    }
    <# -----------------------------------------
     GetBatchMeanImageComparison
        Get Batch Mean Image Comparison
     ------------------------------------------
     Usage: $this.GetBatchMeanImageComparison()
    ----------------------------------------- #>
    [void]Getbatchwarpkeys(){
        #
        $taskname = 'batchwarpkeys'
        $dpath = $this.sample.basepath
        $rpath = '\\' + $this.sample.project_data.fwpath
        $this.getmodulename()
        #
        if ($this.all){
            $pythontask = $this.getpythontask($dpath, $rpath)
        } else{
            $batchslides = $this.sample.batchslides.slideid -join '|'
            $pythontask = $this.getpythontask($dpath, $rpath, $batchslides)
        }
        #
        $this.runpythontask($taskname, $pythontask)
        $this.silentcleanup()
        #
    }
    #
    [string]getpythontask($dpath, $rpath){
        #
        $pythontask = $this.pythonmodulename, $dpath, `
        '--shardedim3root',  $rpath, `
        '--flatfield-file',  $this.sample.pybatchflatfieldfullpath(), `
        '--octets-only --noGPU', $this.buildpyopts('cohort') -join ' '
        #
        return $pythontask
        #
    }
    #
    [string]getpythontask($dpath, $rpath, $batchslides){
        #
        $pythontask = $this.pythonmodulename, $dpath, `
        '--shardedim3root',  $rpath, `
        '--sampleregex',  $batchslides, `
        '--flatfield-file',  $this.sample.pybatchflatfieldfullpath(), `
        '--octets-only --noGPU', $this.buildpyopts('cohort') -join ' '
       #
       return $pythontask
       #
    }
    #
    [void]getmodulename(){
        $this.pythonmodulename = 'warpingcohort'
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