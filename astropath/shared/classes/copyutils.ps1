<# -------------------------------------------
 copyutils
 created by: Benjamin Green - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 copy files fast with different methods
 -------------------------------------------#>
class copyutils{
    #
    copyutils(){}
    #
    [void]copy([string]$sor, [string]$des){
        xcopy $sor, $des /q /y /z /j /v | Out-Null
    }
    #
    [void]copy([string]$sor, [string]$des, [array]$filespec){
        if ($filespec -match '\*'){
            robocopy $sor $des -r:3 -w:3 -np -E -mt:1 | out-null
        } else {
            $filespec = $filespec | foreach-object {'*' + $_}
            robocopy $sor $des $filespec -r:3 -w:3 -np -E -mt:1 | out-null
        }
       # $this.verifyChecksum($sor, $des)
    }
    #
    [void]copy([string]$sor, [string]$des, [array]$filespec, [int]$threads){
        if ($filespec -match '\*'){
            robocopy $sor $des -r:3 -w:3 -np -E -mt:$threads | out-null
        } else {
            $filespec = $filespec | foreach-object {'*' + $_}
            robocopy $sor $des $filespec -r:3 -w:3 -np -E -mt:$threads | out-null
        }
       # $this.verifyChecksum($sor, $des)
    }
    #
    [void]copy([string]$sor, [string]$des, [array]$filespec, [int]$threads, [string]$logfile){
        if ($filespec -match '\*'){
           $output = robocopy $sor $des -r:3 -w:3 -np -E -mt:$threads -log:$logfile
        } else {
           $filespec = $filespec | foreach-object {'*' + $_}
           $output = robocopy $sor $des $filespec -r:3 -w:3 -np -E -mt:$threads -log:$logfile
        }
    }
    #
    [system.object]listfiles([string]$sor, [array]$filespec){
        $sor = $sor + '\*'
        if ($filespec -match '\*'){
            $files = gci $sor -Recurse 
        } else {
            $filespec = $filespec | foreach-object {'*' + $_}
            $files = gci $sor -Include  $filespec -Recurse
        }
        return $files
    }
    #
    [void]verifyChecksum([string]$sor, [string]$des){
        $sourcehash = Get-ChildItem -Path ($sor+'\*') -Recurse | Get-FileHash -Algorithm MD5
        $destinationhash = Get-ChildItem -Path ($des+'\*') -Recurse | Get-FileHash -Algorithm MD5
        $comparison = Compare-Object -ReferenceObject $sourcehash -DifferenceObject $destinationhash -Property Hash |
         Where-Object -FilterScript {$_.SideIndicator -eq '<='}
        if ($comparison -ne $null) {
            foreach ($file in $comparison) {
                $tempsor = ($sourcehash -match $file.Hash).Path
                xcopy $tempsor, $des /q /y /z /j /v | Out-Null
            }
        }
    }
}