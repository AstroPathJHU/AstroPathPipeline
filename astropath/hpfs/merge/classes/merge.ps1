<#
--------------------------------------------------------
merge
Created By: Andrew Jorquera
Last Edit: 12/10/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[array]: the 3 part array of project, slideid, process loc
    E.g. @('7','M18_1','\\bki08\e$')
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [merge]::new($task, $sample)
       $a.runmerge()
--------------------------------------------------------
#>
Class merge : moduletools {
    #
    merge([array]$task, [launchmodule]$sample) : base ([array]$task, [launchmodule]$sample){
        $this.funclocation = '"' + $PSScriptRoot + '\..\MaSS"'
    }
    <# -----------------------------------------
     RunMerge
     Run merge
     ------------------------------------------
     Usage: $this.RunMerge()
    ----------------------------------------- #>
    [void]RunMerge(){
        $this.GetMerge()
    }
    <# -----------------------------------------
     GetMerge
        Get merge using matlab code
     ------------------------------------------
     Usage: $this.GetMerge()
    ----------------------------------------- #>
    [void]GetMerge(){
        $this.sample.info("started merge")
        $taskname = 'MaSS'
        $matlabtask = ";MaSS('" + $this.sample.informfolder() + "', '" + 
            $this.sample.slideid + "', '" + $this.sample.mergeconfigfile() + "', '" + 
            $this.sample.project.PadLeft(2,'0') + ";" + $this.sample.cohort.PadLeft(2, '0') + "');exit(0);"
        $inp.runmatlabtask($taskname, $matlabtask)
        #
        $taskname = 'CreateImageQAQC'
        $matlabtask = ";CreateImageQAQC('" + $this.sample.informfolder() + "', '" + 
            $this.sample.slideid + "', '" + $this.sample.mergeconfigfile() + "', '" + 
            $this.sample.project.PadLeft(2,'0') + ";" + $this.sample.cohort.PadLeft(2, '0') + "');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask)
        $this.sample.info("finished merge")
    }
    <# -----------------------------------------
     silentcleanup
     silentcleanup
     ------------------------------------------
     Usage: $this.silentcleanup()
    ----------------------------------------- #>
    [void]silentcleanup(){
        #
    }
}
