<#
--------------------------------------------------------
meanimage
Created By: Andrew Jorquera -JHU
Last Edit: 09/08/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[array]: the 3 part array of project, slideid, process loc
    E.g. @('7','M18_1','\\bki08\e$')
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [meanimage]::new($task, $sample)
       $a.runmeanimage()
--------------------------------------------------------
#>
Class meanimage : moduletools {
    #
    meanimage([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.flevel = [FileDownloads]::IM3
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'  
    }
    <# -----------------------------------------
     RunMeanImage
     Run mean image
     ------------------------------------------
     Usage: $this.RunMeanImage()
    ----------------------------------------- #>
    [void]RunMeanImage(){
        $this.DownloadFiles()
        $this.ShredDat()
        $this.GetMeanImage()
        $this.returndata()
        $this.cleanup()
    }
   <# -----------------------------------------
     GetMeanImage
        Get the mean image
     ------------------------------------------
     Usage: $this.GetMeanImage()
    ----------------------------------------- #>
    [void]GetMeanImage(){
        $this.sample.info("started getting mean image")
        $taskname = 'raw2mean'
        $matlabtask = ";raw2mean('"+$this.processvars[1]+"', '"+$this.sample.slideid+"');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
        $this.sample.info("finished getting mean image")
    }
    <# -----------------------------------------
     returndata
     returns data to source path
     ------------------------------------------
     Usage: $this.returndata()
    ----------------------------------------- #>
    [void]returndata(){
        #
        $this.sample.info("Return data started")
        #
		$sor = $this.processvars[1] +'\flat'
		$des = $this.sample.im3folder()
		robocopy $sor $des *.flt -r:3 -w:3 -np -mt:30 |out-null
		robocopy $sor $des *.csv -r:3 -w:3 -np -mt:30 |out-null
        # explcitly delete the flat folder after copy so that the 
		# function can be run "in place"
		# by leaving $taskinput[2] blank
		# also robocopy is a parallel copy tool for large files
		# for single files like this, specify their names and 
		# copy using xcopy
        $this.sample.info("Return data finished")
        #
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
        #
        if ($this.processvars[4]){
            Get-ChildItem -Path $this.processloc -Recurse | Remove-Item -force -recurse
        }
        $this.sample.info("cleanup finished")
        #
    }
}
