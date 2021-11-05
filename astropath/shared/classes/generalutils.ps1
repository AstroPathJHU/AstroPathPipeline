class generalutils : copyutils {
     #
    <# -----------------------------------------
     MergePSCustomObject
     Merge two PS Custom Objects based on a property
     return the left-outer-join in a new PS Custom Object
     ------------------------------------------
     Usage: MergePSCustomObject(d1, d2, property)
    ----------------------------------------- #>
    [PSCustomObject]MergeCustomObject([PSCustomObject]$d1, [PSCustomObject]$d2, [string]$property = ''){
        #
        # get new columns
        #
        $columns_d1 = ($d1 | Get-Member -MemberType NoteProperty).Name
        $columns_d2 = ($d2 | Get-Member -MemberType NoteProperty).Name
        $comparison = Compare-Object -ReferenceObject $columns_d1 `
                                     -DifferenceObject $columns_d2 `
                                     -IncludeEqual
        $columns_to_add = ($comparison | `
                           Where-Object -FilterScript {$_.SideIndicator -eq '=>'} `
                           ).InputObject
        #
        # validate that a merge is feasible
        #
        if (!($property -in $comparison.InputObject)){
            Throw "Can merge on $property"
        }
        if (
            !((Compare-Object `
                -ReferenceObject $d1.$property `
                -DifferenceObject $d2.$property `
                -IncludeEqual -ExcludeDifferent).Count `
                -eq $d1.Count)
        ){
            Throw "Can merge on $property"
        }
        #
        # create a new custom object with columns of $d1 add in new $d2 columns
        #
        $d4 = @()
        #
        # get the values of 
        #
        $d1 | ForEach-Object {
            $c = $_.$property
            $d3 = $d2 | Where-Object -FilterScript {$_.$property -eq $c} | `
                        Select-Object -Property $columns_to_add
            #
            $hash = $this.ConvertObjectHash( $_, $columns_d1)
            $hash = $this.ConvertObjectHash($d3, $columns_to_add, $hash)
            #
            $d4 += $hash
        }
        #
        $d5 = [pscustomobject]$d4
        #
        return $d5
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
    [hashtable]ConvertObjectHash([PSCustomObject] $object,[Object] $columns,[Hashtable] $hash){
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
        #
        if ($PSScriptRoot[0] -ne '\'){
            $root = ('\\' + $env:computername+'\'+$PSScriptRoot) -replace ":", "$"
        } else{
            $root = $PSScriptRoot -replace ":", "$"
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
    #
    <# -----------------------------------------
     createdirs
     create a directory if it does not exist 
     ------------------------------------------
     Usage: $this.createdirs()
    ----------------------------------------- #>   
    [void]CreateDirs($dir){
        if (!(test-path $dir)){
            new-item $dir -itemtype "directory" -EA STOP | Out-NULL
        }
    }
    [void]CreateFile($fpath){
        if (!(test-path $fpath)){
            New-Item -path $fpath -itemtype file -Force -EA Stop | Out-Null
        }
    }
    #
    [void]removedir([string]$dir){
        #
        if (test-path $dir){
            gci $dir -Recurse | Remove-Item -force -recurse
            remove-item $dir -force
        }
        #
    }
    #
    [void]removefile([string]$file){
        #
        if (test-path $file){
            remove-item $file -force -ea Continue
        }
        #
    }
    #
    [void]removefile([string]$folder, [string] $filespec){
        #
        $filespec = '*' + $filespec
        $files = gci ($folder+'\*') -Include  $filespec -Recurse 
        if ($files ){ Remove-Item -force -recurse }
        #
    }
}