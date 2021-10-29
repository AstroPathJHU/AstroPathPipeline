<#
--------------------------------------------------------
warpoctets
Created By: Andrew Jorquera
Last Edit: 10/29/2021
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
    warpoctets([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
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
        $this.DownloadFiles()
        $this.ShredDat()
        $this.GetWarpOctets()
        $this.cleanup()
    }
   <# -----------------------------------------
     GetWarpOctets
        Get the warp octets
     ------------------------------------------
     Usage: $this.GetWarpOctets()
    ----------------------------------------- #>
    [void]GetWarpOctets(){
        $this.sample.info("started warp octets")
        $taskname = 'warpoctets'
        #$dpath = $this.sample.basepath + ' '
        $dpath = '\\bki04\Clinical_Specimen'
        $rpath = $this.processvars[1]
        $flatfield = $this.sample.flatwfolder()
        $pythontask = 'warpingcohort ' + $dpath +
         ' --shardedim3root ' + $rpath +
        # ' --workingdir ' + $this.processvars[0] + '\meanimage' +
         " --flatfield-file " + $flatfield + ' --octets-only'
        $this.runpythontask($taskname, $pythontask)
        $this.sample.info("finished warp octets")
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
        if ($this.processvars[4]){
            Get-ChildItem -Path $this.processloc -Recurse | Remove-Item -force -recurse
        }
        $this.sample.info("cleanup finished")
        #
    }
}
