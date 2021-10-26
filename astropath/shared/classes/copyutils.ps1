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
    copyutiles(){}
    #
    [void]copy([string]$sor, [string]$des){
        xcopy $sor, $des /q /y /z /j /v | Out-Null
    }
    #
    [void]copy([string]$sor, [string]$des, [string]$filespec){
        $filespec = '*' + $filespec
        robocopy $sor $des $filespec -r:3 -w:3 -np -mt:1 | out-null
    }
    #
    [void]copy([string]$sor, [string]$des, [string]$filespec, [int]$threads){
        $filespec = '*' + $filespec
        robocopy $sor $des $filespec -r:3 -w:3 -np -mt:$threads | out-null
    }
    #
}