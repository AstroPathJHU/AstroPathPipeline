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
    [switch]$all = $false
    [array]$batchslides
    #
    batchwarpkeys([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.processloc = $this.sample.basepath + '\flatfield\' + $this.sample.batchID
        $this.sample.createdirs($this.processloc)
        $spdir = $this.sample.mpath + '\warping\octets'
        $this.sample.createdirs($spdir)
    }
    <# -----------------------------------------
     RunBatchMeanImageComparison
     Run batch mean image comparison
     ------------------------------------------
     Usage: $this.RunBatchMeanImageComparison()
    ----------------------------------------- #>
    [void]Runbatchwarpkeys(){
        $this.getslideidregex()
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
        $pythontask = $this.getpythontask($dpath, $rpath)
        #
        $this.runpythontask($taskname, $pythontask)
        $this.silentcleanup()
        #
    }
    #
    [void]getslideidregex(){
        #
        $this.sample.info('selecting samples for sample regex')
        #
        $nbatchslides = @()
        $sid = $this.sample.slideid
        #
        if ($this.all){
            $aslides = $this.sample.importslideids($this.sample.mpath)
            $aslides = $aslides | where-object {$_.Project -match $this.sample.project}
            $slides = $aslides.SlideID
        } else {
            $slides = $this.sample.batchslides.slideid
        }
        #
        foreach ($slide in $slides){
            $this.sample.slideid = $slide
            if ($this.sample.testwarpoctetsfiles()){
                $nbatchslides += $slide
            }
        }
        #
        $this.sample.slideid = $sid
        $this.sample.info(([string]$nbatchslides.length +
             ' sample(s) selected for sampleregex'))
        $this.batchslides = $nbatchslides
        #
    }
    #
    [string]getpythontask($dpath, $rpath){
        #
        $this.sample.info('start find keys')
        #
        $pythontask = (
            $this.pythonmodulename, $dpath, 
            '--shardedim3root',  $rpath, 
            '--sampleregex',  ('"'+($this.batchslides -join '|')+'"'), 
            '--flatfield-file',  $this.sample.pybatchflatfieldfullpath(), 
            '--octets-only --noGPU --no-log',
            '--ignore-dependencies',
            $this.buildpyopts('cohort')
         ) -join ' '
       #
       return $pythontask
        #
        $this.sample.info('finished find keys')
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