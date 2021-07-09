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
# check input parameters
#
Class informinput {
 [string]$stringin
 [string]$abx
 [string]$alg
 [string]$abpath
 [string]$algpath
 [string]$outpath
 [string]$infoutpath
 [string]$image_list_file
 [array]$image_list
 [string]$informpath
 [launchmodule]$sample
 #
 informinput(
    [array]$task,
    [launchmodule]$sample
    ){
        $this.sample = $sample
        $this.abx = $task[2].trim()
        $this.alg = $task[3].trim()
        $this.abpath = $this.sample.phenotypefolder()+'\'+$this.abx
        $this.algpath = $this.sample.basepath+'\tmp_inform_data\Project_Development\'+$this.alg
        $this.outpath = "C:\Users\Public\BatchProcessing"
        $this.infoutpath = $this.outpath+"\"+$this.abx
        $this.informpath = '"'+"C:\Program Files\Akoya\inForm\"+$task[4].trim()+"\inForm"+'"'
        #
        # test paths 
        #
        $aQ = test-path $this.algpath
        if (!$aQ){
            Throw "algorithm not found for:" + $this.algpath
        }
        #
        $tQ = test-path $this.sample.flatwim3folder()
        if (!$tQ){
            Throw "flatw path not found for:" + $this.sample.flatwim3folder()
        }
        <#
        $iQ = test-path $this.informpath
        if (!$iQ){
            Write-Host " WARNING: inform path not found for:" $vers -ForegroundColor Magenta
            $this.ee = 3
        } else {
            Write-Host " InForm version:" $vers -ForegroundColor Yellow   
        }
        #>
     }
 #
 CreateImageList(
    ){
        $p = $this.sample.flatwim3folder()+"\*"
        $this.image_list = gci -Path $p -include *.im3 | % {$_.FullName}
        $this.image_list_file = $this.outpath+"\image_list.tmp"
        Set-Content $this.image_list_file $this.image_list
    }
 #
 CreateOutputDir(
    ){
        $tQ = test-path $this.infoutpath
        if ($tQ){
                remove-item $this.infoutpath -force -Recurse
            }
        New-Item $this.infoutpath -itemtype "directory" | Out-NULL
    }
 #
 RunBatchInForm(
    ){
       $lowlog =  $this.outpath+"\output.log"
       $command = $this.informpath+" -a "+$this.algpath+" -o "+$this.infoutpath+" -i "+$this.image_list_file
       cmd /c "$command" > $lowlog
    }
 #
 ReturnData(
    ){
        $iQ = test-path $this.abpath
        if (!$iQ){
            New-Item $this.abpath -itemtype "directory" | Out-NULL
        }
        #
        $sor = $this.infoutpath
        $sor1 = $sor+'\*'
        #
        # remove legend file
        #
        $sor2 = gci $sor1 -include "*legend.txt" | % {$_.FullName}
        Remove-Item $sor2 -Force
        #
        # remove batch_procedure project and add the algorithm ##############validate##################
        #
        try{
            $sor1 = gci $sor1 -include "*.ifr" | % {$_.FullName}
            Remove-Item $sor1 -Force
        } catch{}
        #
        XCOPY /q /y /z $this.algpath $sor 
        #
        $old_name = $sor+'\'+$this.alg
        $new_name = $sor+'\'+'batch_procedure'+$this.alg.Substring($this.alg.Length-4, 4)
        #
        Rename-Item -LiteralPath $old_name $new_name -Force
        #
        XCOPY /q /y /z $sor $this.abpath | Out-NULL
        Remove-Item $this.outpath -Recurse -Force
    }
 #
 CheckErrors(
    ){
        $o = $this.abpath+"\*"
        $ofiles = gci $o -Include *.txt
        $fQ = $ofiles.Length -eq $this.image_list.Length
        $b = $ofiles | Measure Length -Minimum
        $bQ = $b.Minimum -gt 0kb
    }
 #
}
#