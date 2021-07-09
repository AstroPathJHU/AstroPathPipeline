<#
--------------------------------------------------------
inform_worker
Created By: Benjamin Green
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
Input:
$in[string]: the 4 part comma separated list of dpath, 
    slideid, antibody, and algorithm.
    E.g. "\\bki04\Clinical_Specimen_2,M18_1,CD8,CD8_12.05.2018_highTH.ifr"
$vers[string]: The version number of inform to use 
    (must be after the PerkinElmer to Akoya name switch)
    E.g.: "2.4.8"
--------------------------------------------------------
#>
#
Function inform {
     #
     param($task, $this)
     #
     # parse input
     #
     $inp = [informinput]::new($task, $this)
     #
     $this.info("inForm version: " + $task[4])
     $this.info("Create inForm output location")
     $inp.CreateOutputDir()
     $this.info("Compile image list")
     $inp.CreateImageList()
     $this.info("Launch inForm Batch")
     $inp.RunBatchInForm()
     $this.info("inForm Batch Complete")
     $this.info("Launch Data Transfer")
     $inp.ReturnData()
     $this.info("Data Transfer Complete")
     #
     #return $inp.ee
     #
}
#