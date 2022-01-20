﻿<# -------------------------------------------
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
    <# -----------------------------------------
     isWindows
     check if OS is windows (T) or not (F)
     ------------------------------------------
     Usage: isWindows()
    ----------------------------------------- #>
    [switch]isWindows(){
        #
        try{
            $OS = (Get-WMIObject win32_operatingsystem).name
            return $true
        } catch {
            return $false
        }
        #
    }
    <# -----------------------------------------
     copy
     copy a file from one location to another
     ------------------------------------------
     Input: 
        -sor: source file path (one file)
        -des: destination folder path
     ------------------------------------------
     Usage: copy(sor, des)
    ----------------------------------------- #>
    [void]copy([string]$sor, [string]$des){
        xcopy $sor, $des /q /y /z /j /v | Out-Null
        $this.verifyChecksum($sor, $des, '*', 0)
    }
    <# -----------------------------------------
     copy
     copy multiple files with a filespec or multiple
     file specs to a new location
     ------------------------------------------
     Input: 
        -sor: source folder path
        -des: destination folder path
        - filespec: an array of filespecs to transfer
     ------------------------------------------
     Usage: copy(sor, des, filespec)
    ----------------------------------------- #>
    [void]copy([string]$sor, [string]$des, [array]$filespec){
        $filespeco = $filespec
        if ($filespec -match '\*'){
            robocopy $sor $des -r:3 -w:3 -np -E -mt:1 | out-null
        } else {
            $filespec = $filespec | foreach-object {'*' + $_}
            robocopy $sor $des $filespec -r:3 -w:3 -np -s -mt:1 | out-null
        }
        $this.verifyChecksum($sor, $des, $filespeco, 0)
    }
    <# -----------------------------------------
     copy
     copy multiple files with a filespec or multiple
     file specs to a new location with a specified
     number of threads
     ------------------------------------------
     Input: 
        -sor: source folder path
        -des: destination folder path
        - filespec: an array of filespecs to transfer
        - threads: number of threads to use
     ------------------------------------------
     Usage: copy(sor, des, filespec, threads)
    ----------------------------------------- #>
    [void]copy([string]$sor, [string]$des, [array]$filespec, [int]$threads){
        $filespeco = $filespec
        if ($filespec -match '\*'){
            robocopy $sor $des -r:3 -w:3 -np -E -mt:$threads | out-null
        } else {
            $filespec = $filespec | foreach-object {'*' + $_}
            robocopy $sor $des $filespec -r:3 -w:3 -np -s -mt:$threads | out-null
        }
        $this.verifyChecksum($sor, $des, $filespeco, 0)
    }
    <# -----------------------------------------
     copy
     copy multiple files with a filespec or multiple
     file specs to a new location with a specified
     number of threads specifying a log output location
     ------------------------------------------
     Input: 
        -sor: source folder path
        -des: destination folder path
        - filespec: an array of filespecs to transfer
        - threads: number of threads to use
        - logfile: the logfile location (full path)
     ------------------------------------------
     Usage: copy(sor, des, filespec, threads, logfile)
    ----------------------------------------- #>
    [void]copy([string]$sor, [string]$des, [array]$filespec, [int]$threads, [string]$logfile){
        $filespeco = $filespec
        if ($filespec -match '\*'){
           $output = robocopy $sor $des -r:3 -w:3 -np -E -mt:$threads -log:$logfile
        } else {
           $filespec = $filespec | foreach-object {'*' + $_}
           $output = robocopy $sor $des $filespec -r:3 -w:3 -np -s -mt:$threads -log:$logfile
        }
        $this.verifyChecksum($sor, $des, $filespeco, 0)
    }
    <# -----------------------------------------
     listfiles
     list all files with a filespec or multiple
     file specs in a folder
     ------------------------------------------
     Input: 
        - sor: source folder path
        - filespec: an array of filespecs to transfer
     ------------------------------------------
     Usage: copy(sor, filespec)
    ----------------------------------------- #>
    [system.object]listfiles([string]$sor, [array]$filespec){
        $sor = $sor + '\*'
        if ($filespec -match '\*'){
            $files = gci $sor -Recurse 
        } else {
            $filespec = $filespec | foreach-object {'*' + $_}
            $files = gci $sor -Include  $filespec -Recurse
        }
        if ($files -eq $null) {
            $files = @()
        }
        return $files
    }
    <# -----------------------------------------
     verifyChecksum
     create checksums on files to make sure they
     transferred properly if they do not, try 
     those files again.
     ------------------------------------------
     Input: 
        - sor: source folder path
        - des: destination folder path
        - filespec: an array of filespecs to transfer
     ------------------------------------------
     Usage: copy(sor, des, filespec)
    ----------------------------------------- #>
    [void]verifyChecksum([string]$sor, [string]$des, [array]$filespec, [int]$copycount){
        #
        # get the list of files that were transferred
        #
        if ((Get-Item $sor) -is [System.IO.DirectoryInfo]){
            $sourcefiles = $this.listfiles($sor, $filespec)
            $desfiles = $this.listfiles($des, $filespec)
            $sourcehash = $sourcefiles | Get-FileHash -Algorithm MD5
            $destinationhash = $desfiles | Get-FileHash -Algorithm MD5
        } else {
            $sourcefiles = $sor
            $desfiles = $des + '\' + ($sor -split '\\')[-1]
            $sourcehash = Get-FileHash $sourcefiles -Algorithm MD5
            $destinationhash = Get-FileHash $desfiles -Algorithm MD5
        }
        #
        # catch empty folders
        #
        if ($sourcehash -eq $null) {
            $sourcehash = @()
        }
        if ($destinationhash -eq $null) {
            $destinationhash = @()
        }
        #
        # compare hashes
        #
        $comparison = Compare-Object -ReferenceObject $sourcehash `
                                -DifferenceObject $destinationhash `
                                -Property Hash |
                Where-Object -FilterScript {$_.SideIndicator -eq '<='}
        #
        # copy files that failed
        # call checksum on the particular file to make sure the 
        # second go round went properly, fail on the 5th try.
        #
        if ($comparison -ne $null) {
            foreach ($file in $comparison) {
                $tempsor = ($sourcehash -match $file.Hash).Path
                if ($copycount -gt 5){
                    Throw ('failed to copy ' + $tempsor)
                }
                xcopy $tempsor, $des /q /y /z /j /v | Out-Null
                $copycount += 1
                $this.verifyChecksum($tempsor, $des, '*', $copycount)
            }
        }
        #
    }
}