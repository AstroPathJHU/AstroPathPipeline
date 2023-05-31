<#
--------------------------------------------------------
warpoctets
Created By: Andrew Jorquera
Last Edit: 11/1/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[array]: the 3 part array of project, slideid, process loc
    E.g. @('7','M18_1','\\bki08\e$')
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [warpoctets]::new($task, $sample)
       $a.runwarpoctets()
--------------------------------------------------------
#>
Class warpoctets : moduletools {
    #
    [string]$pytype = 'sample'
    #
    warpoctets([hashtable]$task,[launchmodule]$sample) : base ([hashtable]$task,[launchmodule]$sample){
        $this.flevel = [FileDownloads]::IM3
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'  
    }
    <# -----------------------------------------
     RunMeanImage
     Run warp octets
     ------------------------------------------
     Usage: $this.RunMeanImage()
    ----------------------------------------- #>
    [void]RunWarpOctets(){
        $this.sample.CreateNewDirs($this.sample.warpoctetsfolder())
        $this.DownloadFiles()
        $this.ShredDat()
        $this.GetWarpOctets()
        $this.cleanup()
        $this.datavalidation()
    }
   <# -----------------------------------------
     GetWarpOctets
        Get the warp octets
     ------------------------------------------
     Usage: $this.GetWarpOctets()
    ----------------------------------------- #>
    [void]GetWarpOctets(){
        $this.sample.info("started warp octets")
        $this.getmodulename()
        $taskname = $this.pythonmodulename
        #
        $dpath = $this.sample.basepath
        $rpath = $this.processvars[1]
        $pythontask = $this.('getpythontask' + $this.pytype)($dpath, $rpath)
        #
        $this.runpythontask($taskname, $pythontask)
        $this.sample.info("finished warp octets")
    }
    #
    [string]getpythontasksample($dpath, $rpath){
        $globalargs = $this.buildpyopts()
        $pythontask = ($this.pythonmodulename,
            $dpath,
            $this.sample.slideid,
            '--shardedim3root',  $rpath, 
            '--flatfield-file',  $this.sample.pybatchflatfieldfullpath(), 
            $this.gpuopt(), '--no-log', $globalargs
         ) -join ' '
        #
        return $pythontask
    }
    #
    [string]getpythontaskcohort($dpath, $rpath){
        $pythontask = $this.pythonmodulename, $dpath, `
         '--shardedim3root',  $rpath, `
         '--sampleregex',  $this.sample.slideid, `
         '--flatfield-file',  $this.sample.pybatchflatfieldfullpath(), `
         '--octets-only', $this.gpuopt(), $this.buildpyopts() -join ' '
        #
        return $pythontask
    }
    #
    [void]getmodulename(){
        $this.pythonmodulename = ('warping', $this.pytype -join '')
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
        if ($this.processvars[4]){
            $this.sample.removedir($this.processloc)
        }
        #
    }
    <# -----------------------------------------
     datavalidation
     Validation of output data
     ------------------------------------------
     Usage: $this.datavalidation()
    ----------------------------------------- #>
    [void]datavalidation(){
        if (!$this.sample.testwarpoctets()){
            throw 'Output files are not correct'
        }
    }
}
