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
param ([Parameter(Position=0)][string] $in = '',
       [Parameter(Position=1)][string] $vers = '')
#
# check input parameters
#
if (
    !($PSBoundParameters.ContainsKey('in')) -OR 
    !($PSBoundParameters.ContainsKey('vers'))
    ) { 
    Write-Host "Usage: inform_worker in string, version `r"; 
    return
    }
#
Class informinput {
 [string]$stringin
 [array]$arrayin
 [string]$dpath
 [string]$slideid
 [string]$abx
 [string]$alg
 [string]$basepath
 [string]$fwpath
 [string]$abpath
 [string]$algpath
 [int]$ee
 [string]$outpath
 [string]$infoutpath
 [string]$image_list_file
 [array]$image_list
 [string]$informpath
 #
 informinput(
    [string]$in, [string]$vers
     ){
        $this.stringin = $in -replace(' '. '')
        $this.arrayin = $this.stringin -split ','
        $this.dpath = $this.arrayin[0]
        $this.slideid = $this.arrayin[1]
        $this.abx = $this.arrayin[2]
        $this.alg = $this.arrayin[3]
        $this.basepath= $this.dpath+'\'+$this.slideid
        $this.fwpath = $this.basepath+'\im3\flatw'
        $this.abpath = $this.basepath+'\inform_data\Phenotyped\'+$this.abx
        $this.algpath = $this.dpath+'\tmp_inform_data\Project_Development\'+$this.alg
        $this.outpath = "C:\Users\Public\BatchProcessing"
        $this.infoutpath = $this.outpath+"\"+$this.abx
        $this.informpath = '"'+"C:\Program Files\Akoya\inForm\"+$vers+"\inForm"+'"'
        $this.ee = 0
        #
        # test paths 
        #
        $aQ = test-path $this.algpath
        if (!$aQ){
            Write-Host " WARNING: algorithm not found for:" $this.arrayin -ForegroundColor Magenta
            $this.ee = 2
        }
        #
        $tQ = test-path $this.fwpath
        if (!$tQ){
            Write-Host " WARNING: flatw path not found for:" $this.arrayin -ForegroundColor Magenta
            $this.ee = 1
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
        $p = $this.fwpath+"\*"
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
       $log =  $this.outpath+"\output.log"
       $command = $this.informpath+" -a "+$this.algpath+" -o "+$this.infoutpath+" -i "+$this.image_list_file
       cmd /c "$command" > $log
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
        # remove batch_procedure project and add the algorithm
        #
        $sor1 = gci $sor1 -include "*.ifr" | % {$_.FullName}
        Remove-Item $sor1 -Force
        XCOPY /q /y /z $this.algpath $sor 
        #
        $old_name = $sor+'\'+$this.alg
        $new_name = $sor+'\'+'batch_procedure'+$this.alg.Substring($this.alg.Length-4, 4)
        #
        Rename-Item -LiteralPath $old_name $new_name -Force

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
Function inform_worker {
     #
     param($stringin, $vers)
     #
     # parse input
     #
     $inp = [informinput]::new($stringin, $vers)
     if (!($inp.ee -eq 0)){
        return $inp.ee
     }
     #
     Write-Host "." -ForegroundColor Yellow
     Write-Host " InForm version:" $vers -ForegroundColor Yellow
     Write-Host " Create inForm output location" -ForegroundColor Yellow
     $inp.CreateOutputDir()
     Write-Host " Compile image list" -ForegroundColor Yellow
     $inp.CreateImageList()
     Write-Host " Launch inForm Batch" -ForegroundColor Yellow
     $inp.RunBatchInForm()
     Write-Host " inForm Batch Complete" -ForegroundColor Yellow
     Write-Host " Launch Data Transfer" -ForegroundColor Yellow
     $inp.ReturnData()
     Write-Host " Data Transfer Complete" -ForegroundColor Yellow
     Write-Host "." -ForegroundColor Yellow
     #
     return $inp.ee
     #
}
#
inform_worker $in $vers