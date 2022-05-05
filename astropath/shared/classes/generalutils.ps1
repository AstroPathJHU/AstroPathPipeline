﻿class generalutils : copyutils {
     #
    <# -----------------------------------------
     MergePSCustomObject
     Merge two PS Custom Objects based on a property
     return the left-outer-join in a new PS Custom Object
     ------------------------------------------
     Usage: MergePSCustomObject(d1, d2, property)
    ----------------------------------------- #>
    [PSCustomObject]MergeCustomObject([PSCustomObject]$d1,
        [PSCustomObject]$d2, [string]$property = ''){
        #
        # get new columns
        #
        $columns_d1 = ($d1 | Get-Member -MemberType NoteProperty).Name
        $columns_d2 = ($d2 | Get-Member -MemberType NoteProperty).Name
        $comparison = Compare-Object -ReferenceObject $columns_d1 `
                                     -DifferenceObject $columns_d2 `
                                     -IncludeEqual
        $columns_to_add = ($comparison | 
                           Where-Object -FilterScript {$_.SideIndicator -eq '=>'}
                           ).InputObject
        #
        # validate that a merge is feasible
        #
        if (!($property -in $comparison.InputObject)){
            Throw "Can't merge on $property"
        }
        #
        # get matching values of $property to merge on
        #
        $mergeitems = (Compare-Object `
          -ReferenceObject $d1.$property `
          -DifferenceObject $d2.$property `
          -IncludeEqual -ExcludeDifferent).InputObject
        #
        # create a new custom object with columns of $d1 add in new $d2 columns
        #
        $d4 = @()
        #
        # get the values of 
        #
        foreach ($mergeitem in $mergeitems){
            $d6 = $d1 | Where-Object -FilterScript {$_.$property-eq $mergeitem} 
            $d3 = $d2 | 
                Where-Object -FilterScript {$_.$property -eq $mergeitem} | 
                Select-Object -Property $columns_to_add
            #
            $hash = $this.ConvertObjectHash($d6, $columns_d1)
            $hash = $this.ConvertObjectHash($d3, $columns_to_add, $hash)
            #
            $d4 += new-object psobject -Property $hash
        }
        #
        #
        return $d4
        #
    }
    <# -----------------------------------------
     ConvertObjectHash
     Convert a PSCutomObject to a Hash table
     ------------------------------------------
     Input: 
        -object: object to convert
        [-columns]: columns of the object to use
     ------------------------------------------
     Usage: ConvertObjectHash(object, [columns])
    ----------------------------------------- #>
    [hashtable]ConvertObjectHash([PSCustomObject] $object){
        #
        $columns = ($object | Get-Member -MemberType NoteProperty).Name
        #
        $hash = @{}
        #
        foreach($p in $columns){
            $hash.Add($p, $object.$p)
        }
        #
        return $hash
        #
    }
    #
    [hashtable]ConvertObjectHash([PSCustomObject] $object,[Object] $columns){
        #
        $hash = @{}
        #
        foreach($p in $columns){
            $hash.Add($p, $object.$p)
        }
        #
        return $hash
        #
    }
    #
    [hashtable]ConvertObjectHash([PSCustomObject] $object,
        [Object] $columns,[Hashtable] $hash){
        #
        foreach($p in $columns){
            $hash.Add($p, $object.$p)
        }
        #
        return $hash
        #
    }
    #
    [string]defRoot(){
        ##
        $r = $PSScriptRoot -replace( '/', '\')
        if ($r[0] -ne '\'){
            $root = ('\\' + $env:computername+'\'+$r) -replace ":", "$"
        } else{
            $root = $r -replace ":", "$"
        }
        #
        return($root)
        #
    }
    #
    [string]defDrive(){
        $root = $this.defRoot()
        $drive = ($root -split '\$')[0] + '$'
        return($drive)
    }
    #
    [string]defServer(){
        #
        $drive = $this.defDrive()
        $server = ($drive -split '\\')[2]
        #
        return($server)
        #
    }
    <# -----------------------------------------
     createdirs
     create a directory if it does not exist 
     ------------------------------------------
     Usage: $this.createdirs()
    ----------------------------------------- #>   
    [void]CreateDirs($dir){
        #
        $dir = $this.CrossPlatformPaths($dir)
        #
        if (!(test-path $dir)){
            new-item $dir -itemtype "directory" -EA STOP | Out-NULL
        }
    }
    #
    [void]CreateNewDirs($dir){
        #
        $dir = $this.CrossPlatformPaths($dir)
        #
        $this.removedir($dir)
        #
        if (!(test-path $dir)){
            new-item $dir -itemtype "directory" -EA STOP | Out-NULL
        }
        #
    }
    #
    [void]CreateFile($fpath){
        #
        $fpath = $this.CrossPlatformPaths($fpath)
        #
        $this.createDirs((Split-Path $fpath))
        #
        if (!(test-path $fpath)){
            New-Item -path $fpath -itemtype file -Force -EA Stop | Out-Null
        }
        #
    }
    #
    [void]removedir([string]$dir){
        #
        $dir = $this.CrossPlatformPaths($dir)
        #
        if (test-path -literalpath $dir){
            Get-ChildItem -Directory $dir | Remove-Item -force -Confirm:$false -recurse
            remove-item $dir -force -Confirm:$false -Recurse
        }
        #
    }
    #
    [void]removefile([string]$file){
        #
        $file = $this.CrossPlatformPaths($file)
        #
        if (test-path -literalpath $file){
            remove-item -literalpath $file -force -Confirm:$false -ea Continue
        }
        #
    }
    #
    [void]removefile([string]$folder, [string] $filespec){
        #
        $folder = $this.CrossPlatformPaths($folder)
        #
        $filespec = '*' + $filespec
        $files = Get-ChildItem ($folder+'\*') -Include  $filespec -Recurse 
        if ($files ){ $files | Remove-Item -force -recurse -Confirm:$false}
        #
    }
    #
    [void]renamefile([string]$folder, $sor, $des){
        #
        $folder = $this.CrossPlatformPaths($folder)
        #
        $filespec = '*' + $sor
        $files = Get-ChildItem ($folder+'\*') -Include  $filespec -Recurse 
        #
        $files | Foreach-Object {
            $newname = $_.name -replace $sor, $des   
            Rename-Item $_.fullname $newname -ea stop 
        }
        #
    }
    <# ------------------------------------------
    LastWrite
    ------------------------------------------
    get the last write time for a path or file
    ------------------------------------------ #>
    [DateTime]LastWrite([string]$p){
        #
        $p = $this.CrossPlatformPaths($p)
        #
        if (test-path -literalpath $p){
            return (Get-ChildItem $p).LastWriteTime
        } else {
            return Get-Date
        }
        #
    }
    #
    [PSCustomObject]getstoredtable($table){
        #
        $names = $this.gettablenames($table)
        $storedtable = $table | 
            Select-Object -Property $names
        return $storedtable
        #
    }
    #
    [array]gettablenames($table){
        $names = $table |
         Get-Member -MemberType NoteProperty |
         Select-Object -ExpandProperty Name
        return $names
    }
    #
    [PSCustomObject]changedrows($old, $new){
        $headers = ($new | Get-Member -MemberType NoteProperty).Name
        $cmp = compare-object $old $new -Property $headers
        return $cmp 
    }
    #
    [array]changedprojects($old, $new){
        $table = $this.changedrows($old, $new)
        return $table.project
    }
    #
    [array]changedslide($old, $new){
        $cmp = $this.changedrows($old, $new)
        $table = ($cmp | where-object {$_.SideIndicator -match '=>'})
        return $table.slideid
    }
    #
    [array]changedffmodels($old, $new){
        $cmp = $this.changedrows($old, $new)
        $table = ($cmp | where-object {$_.SideIndicator -match '=>'})
        return $table.slideid
    }
    #
    [array]changedcorrmodels($old, $new){
        $cmp = $this.changedrows($old, $new)
        $table = ($cmp | where-object {$_.SideIndicator -match '=>'})
        return $table.slideid
    }
    #
    [array]changedmicomp($old, $new){
        $cmp = $this.changedrows($old, $new)
        $table = ($cmp | where-object {$_.SideIndicator -match '=>'})
        return $table.slideid
    }
    <# -----------------------------------------
     GetCreds
     puts credentials in a string format
     ------------------------------------------
     Usage: $this.GetCreds()
    ----------------------------------------- 
    [array]GetCreds([Pscredential]$login){
        #
        $username = $login.UserName
        $password = $login.GetNetworkCredential().Password
        return @($username, $password)
        #
    }
    #>
}