<#
--------------------------------------------------------
vmcomponentinform
Created By:  Andrew Jorquera
Last Edit: 12/13/2022
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[hashtable]: must contain slideid, 
    antibody, algorithm, and inform version to use
    E.g. @('\\bki04\Clinical_Specimen_2','M18_1','component.ifr','2.4.8.)
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [vmcomponentinform]::new($task, $sample)
       $a.RunBatchInForm()
--------------------------------------------------------
#>
Class vmcomponentinform : vminform {
    #
    vmcomponentinform([hashtable]$task,[launchmodule]$sample) : base ([hashtable]$task, [launchmodule]$sample) {
        #
        $this.needscomponent = $true
        #
    }
    #
    <# -----------------------------------------
     RunVMComponentinForm
     Run the virtual machine inform process 
     ------------------------------------------
     Usage: $this.RunVMComponentinForm()
    ----------------------------------------- #>
    [void]RunVMComponentinForm(){
        $this.runvminform()
    }
}
#