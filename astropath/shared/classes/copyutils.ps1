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
    <# -----------------------------------------
     isWindows
     check if OS is windows (T) or not (F)
     ------------------------------------------
     Usage: isWindows()
    ----------------------------------------- #>
    [switch]isWindows(){
        #
        try{
            (Get-WMIObject win32_operatingsystem).name | out-null
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
        #
        $this.createdirs($des)
        #
        if ($this.isWindows()){
            xcopy $sor $des /q /y /z /j /v | Out-Null
        } else {
            $this.lxcopy($sor, $des)
        }
        #    
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
        if ($this.isWindows()){
            if ($filespec -match '\*'){
                robocopy $sor $des -r:3 -w:3 -np -E -mt:1 | out-null
            } else {
                $filespec = $filespec | foreach-object {'*' + $_}
                robocopy $sor $des $filespec -r:3 -w:3 -np -s -mt:1 | out-null
            }
        } else {
            $this.lxcopy($sor, $des, $filespec)
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
        if ($this.isWindows()){
            if ($filespec -match '\*'){
                robocopy $sor $des -r:3 -w:3 -np -E -mt:$threads | out-null
            } else {
                $filespec = $filespec | foreach-object {'*' + $_}
                robocopy $sor $des $filespec -r:3 -w:3 -np -s -mt:$threads | out-null
            }
        } else {
            $this.lxcopy($sor, $des, $filespec)
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
        if ($this.isWindows()){
            if ($filespec -match '\*'){
               robocopy $sor $des -r:3 -w:3 -np -E -mt:$threads -log:$logfile | out-null
            } else {
               $filespec = $filespec | foreach-object {'*' + $_}
               robocopy $sor $des $filespec -r:3 -w:3 -np -s -mt:$threads -log:$logfile  | out-null
            }
        } else {
            $this.lxcopy($sor, $des, $filespec)
        }
        $this.verifyChecksum($sor, $des, $filespeco, 0)
    }
    <# -----------------------------------------
     lxcopy
     copy a file from one location to another
     on linux
     ------------------------------------------
     Input: 
        -sor: source file path (one file)
        -des: destination folder path
     ------------------------------------------
     Usage: lxcopy(sor, des)
    ----------------------------------------- #>
    [void]lxcopy($sor, $des){
        #
        $sor1 = $sor -replace '\\', '/'
        $des1 = $des -replace '\\', '/'
        mkdir -p $des1
        cp $sor1 $des1 -r
        #
    }
    <# -----------------------------------------
     lxcopy
     copy a file from one location to another
     on linux
     ------------------------------------------
     Input: 
        -sor: source file path (one file)
        -des: destination folder path
     ------------------------------------------
     Usage: lxcopy(sor, des)
    ----------------------------------------- #>
    [void]lxcopy($sor, $des, $filespec){
        #
        $sor1 = ($sor -replace '\\', '/') + '/'
        $des1 = $des -replace '\\', '/'
        mkdir -p $des1
        #
        if (!($filespec -match '\*')){
            $filespec = $filespec | foreach-object {'*' + $_}
        }
        #
        $filespec | ForEach-Object{
            cp ($sor1+$_) $des1 -r
        }
        #
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
            $files = get-ChildItem $sor -Recurse 
        } else {
            $filespec = $filespec | foreach-object {'*' + $_}
            $files = get-ChildItem $sor -Include  $filespec -Recurse
        }
        if (!$files) {
            $files = @()
        }
        return $files
    }
    <#------------------------------------------
    handlebrackets
    -------------------------------------------#>
    [string]handlebrackets([string]$fname){
        $fname = $fname.replace('[','`[').replace(']','`]')
        return $fname
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
            #
            $sourcefiles = $this.listfiles($sor, $filespec)
            $desfiles = $this.listfiles($des, $filespec)
            #
            #
            if ($global:PSVersionTable.PSVersion -lt 7){
                $sourcehash = $this.FileHasher($sourcefiles)
                $destinationhash = $this.FileHasher($desfiles)
            } else {
                $sourcehash = $this.FileHasher($sourcefiles, 7)
                $destinationhash = $this.FileHasher($desfiles, 7)
            }
        } else {
            #
            $sourcefiles = $this.handlebrackets($sor)
            $desfiles = $this.handlebrackets($des + '\' + (Split-Path $sor -Leaf))
            #
            $sourcehash = $this.FileHasher($sourcefiles, 7, $true)
            $destinationhash = $this.FileHasher($desfiles, 7, $true)
            #
        }
        #
        # catch empty folders
        #
        if ($sourcehash.count -eq 0) {
            $sourcehash = @{}
            $sourcehash.('tmp') = 'tmp'
        }
        if ($destinationhash.count -eq 0) {
            $destinationhash = @{}
            $destinationhash.('tmp') = 'tmp'
        }
        #
        $comparison = Compare-Object -ReferenceObject $($sourcehash.Values) `
                                -DifferenceObject $($destinationhash.Values) `
                        |
                Where-Object -FilterScript {$_.SideIndicator -eq '<='}
        #
        # copy files that failed
        # call checksum on the particular file to make sure the 
        # second go round went properly, fail on the 5th try.
        #
        if ($comparison) {
            #
            foreach ($file in $comparison) {
                $tempsor = ($sourcehash.GetEnumerator() | 
                    Where-Object {$_.Value -contains $file.InputObject}).Key
                #
                if ($copycount -ge 50){
                    Throw ('failed to copy ' + $tempsor + '. N tries:' + $copycount)
                }
                #
                $this.createdirs($des)
                #
                if ($this.isWindows()){
                    xcopy $tempsor $des /q /y /z /j /v | Out-Null
                } else {
                    $this.lxcopy($sor, $des)
                }
                #
                $copycount += 1
                $this.verifyChecksum($tempsor, $des, '*', $copycount)
            }
        }
        #
    }
    <#-----------------------------------------
    FileHasher
    get the file hash if the version is 7.0 or higher use parallel
    processing.
    Edited from 'Get-FileHash' source code
    -----------------------------------------#>
    [System.Collections.Concurrent.ConcurrentDictionary[string,object]]FileHasher($filelist, [int]$v){
        #
        [System.Collections.Concurrent.ConcurrentDictionary[string,object]]$hashes = @{}
        #
        $filelist | foreach-Object -Parallel{
            #
            $hcopy = $using:hashes
            $Algorithm="MD5"
            $hasherType = "System.Security.Cryptography.${Algorithm}CryptoServiceProvider" -as [Type]
            if ($hasherType) {
                $hasher = $hasherType::New()
            }
            #
            if(Test-Path -LiteralPath $_ -PathType Container) {
                continue
            }
            #
            if (!(Test-path $_)){
                continue
            }
            #
            try{
                [system.io.stream]$stream = [system.io.file]::OpenRead($_.FullName)
                [Byte[]] $computedHash = $hasher.ComputeHash($stream)
                [string] $hash = [BitConverter]::ToString($computedHash) -replace '-',''
                $hcopy.($_.FullName) = $hash
            } catch {
                    Throw $_.Exception.Message
            } finally {
                if($stream)
                {
                    $stream.Dispose()
                }
            }
        -ThrottleLimit 20
        }
        #
        return ($hashes)
        #
    }
    <#-----------------------------------------
    FileHasher
    get the file hash if the version is 7.0
     or higher use parallel processing.
    Edited from 'Get-FileHash' source code
    -----------------------------------------#>
    [hashtable]FileHasher($filelist){
        #
        if ($filelist.Count -eq 1){
            $filehash = $this.FileHasher($filelist, 7, $true)
            return $filehash
        }
        #
        $filehash = @{}
        $filehash1 = $filelist | Get-FileHash -Algorithm MD5
        $filehash1 | ForEach-Object{
            $filehash.($_.Path) = $_.Hash
        }
        #
        return $filehash
        #
    }

    [hashtable]FileHasher($file, [int]$v, $singlefile){
        #
        $filehash = @{}
        if (test-path $file){
            $filehash1 = Get-FileHash $file -Algorithm MD5
            $filehash.($file) = $filehash1.Hash
        }  
        return $filehash
        #
    }
}