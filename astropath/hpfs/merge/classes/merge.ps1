<#
--------------------------------------------------------
merge
Benjamin Green, Andrew Jorquera
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
    merge([hashtable]$task, [launchmodule]$sample) : base ([hashtable]$task, [launchmodule]$sample){
        $this.funclocation = '"' + $PSScriptRoot + '\..\funcs\MaSS"'
        if ($this.processvars[4]){
            $this.sample.createdirs($this.processloc)
        }
    }
    <# -----------------------------------------
     RunMerge
     Run merge
     ------------------------------------------
     Usage: $this.RunMerge()
    ----------------------------------------- #>
    [void]RunMerge(){
        $this.getalgorithmlist()
        $this.GetMerge()
        $this.GetQAQC()
        $this.cleanup()
    }
    #
    [void]getalgorithmlist(){
        #
        $this.sample.info('printing algorithms used')
        $algs = $this.sample.getalgorithms()
        $algs | foreach-object {
            $this.sample.setfile(
                $this.sample.mergealgorithmsfile(),
                ($_ + '`r`n' )
            )
        }
        #
        $this.sample.info($algs)
        #
    }
    <# -----------------------------------------
     GetMerge
        Get merge using matlab code
     ------------------------------------------
     Usage: $this.GetMerge()
    ----------------------------------------- #>
    [void]GetMerge(){
        #
        $this.sample.info("started merge")
        $taskname = 'MaSS'
        $matlabtask = ";MaSS('" + $this.sample.informfolder() + "', '" + 
            $this.sample.slideid + "', '" + $this.sample.mergeconfigfile() + "', '" + 
            $this.sample.project.PadLeft(2,'0') + ";" + $this.sample.cohort.PadLeft(2, '0') + ";');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask)
        $this.sample.info("finished merge")
        #
    }
    <# -----------------------------------------
     GetQAQC
        Get QAQC using matlab code
     ------------------------------------------
     Usage: $this.GetQAQC()
    ----------------------------------------- #>
    [void]GetQAQC(){
        #
        $this.sample.info("started image QAQC")
        $taskname = 'CreateImageQAQC'
        $matlabtask = ";CreateImageQAQC('" + $this.sample.informfolder() + "', '" + 
            $this.sample.slideid + "', '" + $this.sample.mergeconfigfile() + "', '" + 
            $this.sample.project.PadLeft(2,'0') + ";" + $this.sample.cohort.PadLeft(2, '0') + ";');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask)
        $this.sample.info("finished image QAQC")
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
    #
    [void]cleanup(){
        #
        $this.silentcleanup()
        #
    }
}
