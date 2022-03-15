<#
--------------------------------------------------------
batchmeanimagecomparison
Benjamin Green, Andrew Jorquera
Last Edit: 02.16.2022
--------------------------------------------------------
#>
class batchwarpkeys : moduletools {
    #
    batchwarpkeys([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.processloc = $this.sample.warpbatchfolder() 
    }
    <# -----------------------------------------
     RunBatchMeanImageComparison
     Run batch mean image comparison
     ------------------------------------------
     Usage: $this.RunBatchMeanImageComparison()
    ----------------------------------------- #>
    [void]Runbatchwarpkeys(){
        #
        $this.getslideidregex('batchwarpkeys')
        $this.getbatchwarpoctets()
        $this.Getbatchwarpkeys()
        $this.datavalidation()
        #
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
        #
    }
    #
    [void]getbatchwarpoctets(){
        #
        if ($this.all){
            return
        }
        #
        $this.sample.info('copying batch octets to working dir')
        $sid = $this.sample.slideid
        #
        $this.batchslides | ForEach-Object{
           $this.sample.slideid = $_
           $this.sample.copy($this.sample.warpoctetsfile(),
            $this.sample.warpbatchoctetsfolder())
        }
        #
        $this.sample.slideid = $sid
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
            $this.buildpyopts('cohort'),
            $this.workingdir()
         ) -join ' '
       #
       return $pythontask
        #
        $this.sample.info('finished find keys')
       #
    }
    #
    [string]workingdir(){
        #
        if ($this.all){
           $warpkeysfolder = $this.sample.warpprojectfolder()
        } else {
            $warpkeysfolder = $this.sample.warpbatchfolder()
        }
        #
        $this.sample.CreateNewDirs(($warpkeysfolder+ '\octets'))
        return ('--workingdir ' + $warpkeysfolder)
        #
    }
    #
    [void]getmodulename(){
        $this.pythonmodulename = 'warpingcohort'
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
        $this.silentcleanup()
        $this.sample.info("cleanup finished")
        #
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
        if (!$this.sample.testbatchwarpkeys()){
            throw 'Output files are not correct'
        }
    }
}