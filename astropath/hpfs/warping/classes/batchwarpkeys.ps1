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
        $this.updateprocessloc()
        $this.getslideidregex('batchwarpkeys')
        $this.getbatchwarpoctets()
        $this.Getbatchwarpkeys()
        $this.datavalidation()
        #
    }
    [void]updateprocessloc(){
        if ($this.all){
            $this.processloc =  $this.sample.warpprojectfolder() 
        }
    }
    <# -----------------------------------------
     GetBatchMeanImageComparison
        Get Batch Mean Image Comparison
     ------------------------------------------
     Usage: $this.GetBatchMeanImageComparison()
    ----------------------------------------- #>
    [void]Getbatchwarpkeys(){
        #
        $this.sample.info('start find keys')
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
        $this.sample.info('finished find keys')
        #
    }
    #
    [void]getbatchwarpoctets(){
        #
        $this.sample.info('copying slide octets to working dir')
        $sid = $this.sample.slideid
        #
        $this.sample.CreateNewDirs(($this.processloc+ '\octets'))
        #
        $this.batchslides | ForEach-Object{
           $this.sample.slideid = $_
           $this.sample.copy($this.sample.warpoctetsfile(),
            ($this.processloc + '\octets'))
        }
        #
        $this.sample.slideid = $sid
        #
    }
    #
    [string]getpythontask($dpath, $rpath){
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
    }
    #
    [string]workingdir(){
        #
        return ('--workingdir ' + $this.processloc)
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