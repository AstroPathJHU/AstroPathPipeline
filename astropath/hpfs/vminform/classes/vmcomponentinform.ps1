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
    [string]$informvers = '2.4.8'
    #
    vmcomponentinform([hashtable]$task,[launchmodule]$sample) : base ([hashtable]$task, [launchmodule]$sample, $true) {
        #
        $this.flevel = [FileDownloads]::FLATWIM3
        #
        $this.sample = $sample
        $this.abx = 'Component'
        $this.alg = 'component_' + $this.sample.BatchID + '.ifr'
        $this.needscomponent = $true
        $this.abpath = $this.sample.phenotypefolder() + '\' + $this.abx
        $this.algpath = $this.sample.basepath +
             '\tmp_inform_data\Project_Development\Component\' + $this.alg
        $this.informoutpath = $this.outpath + "\" + $this.abx + '_' + $this.err
        $this.informpath = '"'+"C:\Program Files\Akoya\inForm\" + 
            $this.informvers + "\inForm.exe"+'"'
        $this.informbatchlog = $this.informoutpath + "\Batch.log"
        $this.processvars[0] = $this.outpath
        $this.processvars[1] = $this.outpath
        $this.processvars[2] = $this.outpath
        #
        if ($this.islocal){
            $this.inputimagepath = $this.sample.flatwim3folder()
        } else {
            $this.inputimagepath = $this.outpath + '\' + $this.sample.slideid + '\im3\flatw'
        }
        $this.processvars += 1
        #
        $this.TestPaths()
        $this.KillinFormProcess()
        #
    }
    #
}
#