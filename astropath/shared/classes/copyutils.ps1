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
    [int]$ntries = 50
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
        if ($env:OS -contains 'Windows_NT'){
            return $true
        }
        return $false
        #
    }
    #
    [string]CrossPlatformPaths($dir){
        #
        if (!$this.isWindows()){
            $dir = $dir -replace '\\', '/'
        } else{
            $dir = $dir -replace '/', '\'
        }
        #
        return $dir
    }
    <# ------------------------------------------
    CheckPath
    ------------------------------------------
    check if a path exists
    ------------------------------------------ #>
    [switch]CheckPath([string]$p){
        #
        $p = $this.CrossPlatformPaths($p)
        #
        if (test-path -literalpath $p){
            return $true
        } else {
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
        try {
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
        }
        catch {
            #Ignore copy exceptions
        }
        finally {
            $this.verifyChecksum($sor, $des, $filespeco, 0)
        }
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
        try {
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
        }
        catch {
            #Ignore copy exceptions
        }
        finally {
            $this.verifyChecksum($sor, $des, $filespeco, 0)
        }
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
        try {
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
        }
        catch {
            #Ignore copy exceptions
        }
        finally {
            $this.verifyChecksum($sor, $des, $filespeco, 0)
        }
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
        if (Test-Path $sor -PathType Container) {
            $this.lxcopy($sor, $des, '*')
            return
        }
        #
        $sor1 = $sor -replace '\\',  '/'
        $des1 = $des -replace '\\', '/'
        #
        if (!(Test-Path $des1)) {
            mkdir -p $des1
        }
        #
        Copy-Item -LiteralPath $sor1 $des1 -r
        #
    }
    <# -----------------------------------------
     lxcopy
     copy a file from one location to another
     on linux
     ------------------------------------------
     Input: 
        -sor: source folder path
        -des: destination folder path
        -filespec: file specifier
     ------------------------------------------
     Usage: lxcopy(sor, des, filespec)
    ----------------------------------------- #>
    [void]lxcopy($sor, $des, $filespec){
        #
        $des1 = $des -replace '\\', '/'
        $sor1 = $sor -replace '\\', '/'
        #
        if (!(Test-Path $des1)) {
            mkdir -p $des1
        }
        #
        $files = $this.listfiles($sor1, $filespec)
        #
        $files | foreach-Object -Parallel { 
            Copy-Item -LiteralPath $_ -r $using:des1 
        } -ThrottleLimit 20
        #
        $gitignore = $sor1 + '/.gitignore'
        if (test-path -LiteralPath $gitignore){
            Copy-Item $gitignore $des1
        }
        #
    }
    #
    [string]uncpaths($path){
        #
        $r = $path -replace( '/', '\')
        if ($r[0] -ne '\'){
            $root = '\\' + $path
        } else{
            $root = $path
        }
        #
        return $root
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
     Usage: listfiles(sor, filespec)
    ----------------------------------------- #>
    [system.object]listfiles([string]$sor, [array]$filespec){
        #
        if ($this.isWindows()) {
            if (!($this.testpathi($sor))){
                return @()
            }
            if ($filespec -match '\*'){
                $files = [system.io.directory]::enumeratefiles($sor, '*.*', 'AllDirectories')  |
                     & {process{[System.IO.FileInfo]$_}}
            } else {
                $filespec = ($filespec | foreach-object {'.*' + $_ + '$'}) -join '|'
                $files = [system.io.directory]::enumeratefiles($sor, '*.*', 'AllDirectories')  |
                     & {process{
                         if ($_ -match $filespec){
                            [System.IO.FileInfo]$_
                         }
                    }}
            }
        }
        else {
            $files = $this.lxlistfiles($sor, $filespec)
        }
        #
        if (!$files) {
            $files = @()
        }
        return $files
    }
    <# -----------------------------------------
     lxlistfiles
     list all files with a filespec or multiple
     file specs in a folder in a linux os
     ------------------------------------------
     Input: 
        - sor: source folder path
        - filespec: an array of filespecs to transfer
     ------------------------------------------
     Usage: lxlistfiles(sor, filespec)
    ----------------------------------------- #>
    [system.object]lxlistfiles([string]$sor, [array]$filespec){
        #
        if (!$sor){
            return @()
        }
        #
        if ($filespec -match '\*'){
            $files = Get-ChildItem $sor
        } else {
            $files = ([array](Get-ChildItem $sor)) | Where-Object {$_.name -match $filespec}
        }
        return $files
    }
    #
    [system.object]fastlistfiles([string]$sor, [array]$filespec){
        #
        if (!($this.testpathi($sor))){
            return @()
        }
        #
        $files = cmd /c "dir /a-d /b /s $sor"
        $sor = $sor + '\*'
        #
        if (!($filespec -match '\*')){
            $filespec = ($filespec | foreach-object {'.*' + $_ + '$'}) -join '|'
            $files = $files | & { process { 
                if ($_ -match $filespec){ $_ }
            }}
        }
        #
        if (!$files) {
            $files = @()
        }
        return $files
        #
    }
    #
    [int]countfiles([string]$sor, [array]$filespec){
        #
        $cnt = 0
        if (!($this.testpathi($sor))){
            return $cnt
        }
        $filespec | foreach-object {
          $cnt +=  @([System.IO.Directory]::EnumerateFiles(
              $sor,  ('*' + $_ ))).Count 

        }
        return $cnt
        #
    }
    #
    [array]getfullnames([string]$sor, [array]$filespec){
        return ($this.fastlistfiles($sor, $filespec))
    }
    #
    [array]getnames([string]$sor, [array]$filespec){
        return (
            split-path ($this.fastlistfiles($sor, $filespec)) -leaf
        )
    }
    <#------------------------------------------
    handlebrackets
    -------------------------------------------#>
    [string]handlebrackets([string]$fname){
        $fname = $fname.replace('[','`[').replace(']','`]')
        return $fname
    }
    <# -----------------------------------------
    testpaths 
    test a path. If the path fails, check multiple
    times of the next few seconds to try to see 
    if the connection is a result of a bad 
    network connection.
    ----------------------------------------- #> 
    [void]testpath($path){
        #
        $cnt = 0
        while($cnt -lt 4){
            if (!$this.checkpath($path)){
                $cnt += 1
                Start-Sleep 2
            } else {
                break
            }
            #
        }
        #
        if ($cnt -eq 4){
            Throw ('path could not be found:' + $path)
        }
        #
    }
    #
    [void]testpath($path, $create){
        #
        $cnt = 0
        while($cnt -lt 4){
            if (!$this.checkpath($path)){
                $cnt += 1
                Start-Sleep 2
            } else {
                break
            }
            #
        }
        #
        $this.createdirs($path)
        #
    }
    <# -----------------------------------------
    testpathi 
    test a path and return a logical. platform 
    depedent.
    ----------------------------------------- #> 
    [switch]testpathi($path){
        if (!$this.isWindows()){
            return (test-path $path)
        } else {
            return ([System.IO.Directory]::Exists($path))
        }
    }
    <# -----------------------------------------
    testpathi 
    test a path and return a logical. platform 
    depedent.
    ----------------------------------------- #> 
    [switch]testfilei($path){
        if (!$this.isWindows()){
            return (test-path $path)
        } else {
            return ([System.IO.File]::Exists($path))
        }
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
        - copycount: an int that indicates 
            the number of times a file has 
            attempted to be copied.
     ------------------------------------------
     Usage: verifyChecksum(sor, des, filespec, copycount)
    ----------------------------------------- #>
    [void]verifyChecksum([string]$sor, [string]$des, 
        [array]$filespec, [int]$copycount){
        #
        $this.testpath($sor)
        $this.testpath($des, $true)
        #
        $missingfiles = $this.checknfiles($sor, $des, $filespec)
        $this.retrycopyloop($missingfiles, $copycount, $sor, $des)
        #
        [array]$hashes = $this.FileHashHandler($sor, $des, $filespec)
        $comparison = $this.comparehashes($hashes[0], $hashes[1])
        $this.retrycopyloop($comparison, $copycount, $sor, $des)
        #
    }
    <#
        check n files 
    #>
    [array]checknfiles($sor, $des, $filespec){
        #
        $missingfiles = @()
        #
        if ((Get-Item $sor) -is [System.IO.DirectoryInfo]){
            #
            $sourcefiles = $this.listfiles($sor, $filespec)
            $desfiles = $this.listfiles($des, $filespec)
            if ($desfiles.length -eq 0) {
                $missingfiles = $sourcefiles.FullName
            }
            else {
                $missingfiles = ($sourcefiles | 
                    & { process {
                        if ( $desfiles.name -notcontains $_.Name) { $_ }
                    }}
                    ).FullName
            }
        } else {
            $sourcefiles = $sor
            $desfiles = $des + '\' + (Split-Path $sor -Leaf)
            #
            if (!(test-path -literalpath $desfiles)){
                $missingfiles += $sourcefiles
            }
            #
        }
        #
        return $missingfiles
        #
    }
    <# -----------------------------------------
    filehashhandler
    handle the file hasher depending on if the 
    input sorce was a file or a destination
    -----------------------------------------
    Input: 
        - sor: source folder path
        - des: destination folder path
        - filespec: an array of filespecs to transfer
    ----------------------------------------- #>
    [array]FileHashHandler($sor, $des, $filespec){
        #
        if ((Get-Item $sor) -is [System.IO.DirectoryInfo]){
            #
            $sourcefiles = $this.listfiles($sor, $filespec)
            $desfiles = $this.listfiles($des, $filespec)
            #
            if ($global:PSVersionTable.PSVersion -lt 7){
                $sourcehash = $this.FileHasher($sourcefiles)
                $destinationhash = $this.FileHasher($desfiles)
            } else {
                $sourcehash = $this.FileHasher($sourcefiles, 7)
                $destinationhash = $this.FileHasher($desfiles, 7)
            }
            #
        } else {
            #
            $sourcefiles = $sor
            $desfiles = $des + '\' + (Split-Path $sor -Leaf)
            #
            $sourcehash = $this.FileHasher($sourcefiles, 7, $true)
            $destinationhash = $this.FileHasher($desfiles, 7, $true)
            #
        }
        #
        # catch empty
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
        return @($sourcehash, $destinationhash)
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
        $job = $filelist | foreach-Object -AsJob -Parallel {
            #
            $hcopy = $using:hashes
            $Algorithm="MD5"
            $hasherType = "System.Security.Cryptography.${Algorithm}CryptoServiceProvider" -as [Type]
            if ($hasherType) {
                $hasher = $hasherType::New()
            }
            #
            if(Test-Path -LiteralPath $_.FullName -PathType Container) {
                continue
            }
            #
            if (!(Test-path -LiteralPath $_.FullName)){
                continue
            }
            #
            try{
                [system.io.stream]$stream = [system.io.file]::OpenRead($_.FullName)
                [Byte[]] $computedHash = $hasher.ComputeHash($stream)
                [string] $hash = [BitConverter]::ToString($computedHash) -replace '-',''
                $cnt = 0
                while(!($hcopy.TryAdd($_.FullName, $hash))){
                    if ($cnt -gt 4){
                        break
                    }
                }
                #
            } catch {
                Throw $_.Exception.Message
            } finally {
                if($stream)
                {
                    $stream.Dispose()
                }
            } 
        } -ThrottleLimit 20
        #
        wait-job $job.id
        receive-job $job.id -ea stop
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
    #
    [hashtable]FileHasher($file, [int]$v, $singlefile){
        #
        $filehash = @{}
        if (test-path -LiteralPath $file){
            $filehash1 = Get-FileHash $file -Algorithm MD5
            $filehash.($file) = $filehash1.Hash
        }  
        return $filehash
        #
    }
    <# -----------------------------------------
    comparehashes
    compare the hash values added and return a
    vector of hashes in input 1 not correct
    or existant in input 2.
    ----------------------------------------- #>
    [array]comparehashes($sourcehash, $destinationhash){
        #
        try{
            $notmatch = Compare-Object -ReferenceObject $sourcehash.values `
                                    -DifferenceObject $destinationhash.values |
                    & { process { if ($_.SideIndicator -eq '<=') {$_}}}
        } catch {
            if ($_.Exception.Message -match 'ReferenceObject'){
                Throw ('source hash values not valid: ' +  $sourcehash.Values)
            } elseif ($_.Exception.Message -match 'DifferenceObject'){
                Throw ('destination hash values not valid: ' +  $destinationhash.Values)
            } else {
                Throw $_.Exception.Message
            }
        }
        #
        if ($notmatch){
            $shashstrings = @()
            #
            $path = split-path $sourcehash.keys[0] 
            #
            foreach ($skey in $sourcehash.keys){
                $file = Split-Path $skey -Leaf
                $shashstrings += $file, $sourcehash.($skey) -join ';'
            }
            #
            $dhashstrings = @()
            #
            foreach ($dkey in $destinationhash.keys){
                $file = Split-Path $dkey -Leaf
                $dhashstrings += $file, $destinationhash.($dkey) -join ';'
            }
            #
            $notmatch = Compare-Object $shashstrings $dhashstrings |
                & {process { if ($_.SideIndicator -eq '<=') {$_}}}
            #
            $notmatch = $notmatch.InputObject | &{ process {
                    return ($path + '\' + ($_ -split ';')[0])
            }}
            #
        }
        #
        return $notmatch
        #
    }
    <# -----------------------------------------
    retrycopyloop
    loop through each comparison object and 
    retry the copy for a provided comparison object 
    -----------------------------------------
    Input: 
        - comparison: the comparison object
        from a compare-object
        - $sourcefile
        - sor: source folder path
        - des: destination folder path
        - filespec: an array of filespecs to transfer
        - copycount: an int that indicates 
            the number of times a file has 
            attempted to be copied.
    ----------------------------------------- #>
    [void]retrycopyloop($notmatch, 
        $copycount, $sor, $des){
        #
        if ($notmatch) {
            foreach ($file in $notmatch) {
                #
                if ($copycount -ge $this.ntries){
                    Throw ('failed to copy ' +
                        $file + ' to ' + $des + '. N tries: ' + $copycount)
                }
                #
                $this.retrycopy($file, $sor, $des, $copycount)
                #
            }
        }
    }
    <#
    retrycopy
    retry copying a specific file. 
    #>
    [void]retrycopy($tempsor, $sor, $des, $copycount){
        #
        Start-Sleep 2
        #
        $this.createdirs($des)
        #
        if ($this.isWindows()){
            xcopy $tempsor $des /q /y /z /j /v | Out-Null
        } else {
            $this.lxcopy($sor, $des)
        }
        #
        $copycount ++
        $this.verifyChecksum($tempsor, $des, '*', $copycount)
        #
    }
    #
    [void]writeoutput($out){
        Write-Host ("$out - " + ($this.getformatdate())) -ForegroundColor Yellow
        $this.writetodailylog($out)
    }
    #
    [void]writetodailylog($out){
        $date = Get-Date -Format "MM.dd.yyyy"
        $dailylogfolder = $dailylog = "C:/Users/Public/astropath/dailylogs/"
        $dailylog = $dailylogfolder + "astropath.$date.log"
        #
        if (!(Test-Path $dailylogfolder)) {
            New-Item $dailylogfolder -ItemType "directory"
        }
        if (!(Test-Path $dailylog)) {
            New-Item $dailylog
        }
        Add-Content $dailylog $out
    }
    #
    [string]getformatdate(){
        return ([string](Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    }
    #
    [string]getformatdate($d){
        return ([string](Get-Date $d -Format "yyyy-MM-dd HH:mm:ss"))
    }
}