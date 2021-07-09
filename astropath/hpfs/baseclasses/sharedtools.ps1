<# -------------------------------------------
 shared_tools
 created by: Benjamin Green - JHU- 10.13.2020

 general functions which may be needed by
 multiple modules
 -------------------------------------------#>
 Class sharedtools{
    [string]$module
    [string]$mpath
    [string]$psroot = $pshome + "\powershell.exe"
     <# -----------------------------------------
      Open-CSVFile
      open a file with error checking
      -------------------------------------------
     Usage:
     Open-CSVFile -fpath
        -fpath: file path to read in
     -------------------------------------------#>
    [PSCustomObject]OpenCSVFile([string] $fpath){
        #
        $cnt = 0
        $Max = 120
        $mxtstring = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
        #
        if(!(test-path $fpath)){
           Throw $fpath + " could not be found"
        }
        #
        $Q = New-Object -TypeName psobject
        #
        do{
            $mxtx = New-Object System.Threading.Mutex($false, $mxtstring)
            try{
                $imxtx = $mxtx.WaitOne(60 * 10)
                if($imxtx){
                    $Q = Import-CSV $fpath -ErrorAction Stop
                    $mxtx.releasemutex()
                    break
                } else {
                    $cnt = $cnt + 1
                    Start-Sleep -s 3
                }
            }catch{
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
            }
        } while($cnt -lt $Max)
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $Max){
           Throw "could not read " + $fpath
        }
        #
        return $Q
        #
    }
    <# -----------------------------------------
      Open-CSVFile
      open a file with error checking
      -------------------------------------------
     Usage:
     Open-CSVFile -fpath
        -fpath: file path to read in
     -------------------------------------------#>
    [Array]GetContent([string] $fpath){
        #
        $cnt = 0
        $Max = 120
        $mxtstring = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
        #
        if(!(test-path $fpath)){
           Throw $fpath + " could not be found"
        }
        #
        $Q = New-Object -TypeName psobject
        #
        do{
            $mxtx = New-Object System.Threading.Mutex($false, $mxtstring)
            try{
                $imxtx = $mxtx.WaitOne(60 * 10)
                if($imxtx){
                    $Q = Get-Content $fpath -ErrorAction Stop
                    $mxtx.releasemutex()
                    break
                } else {
                    $cnt = $cnt + 1
                    Start-Sleep -s 3
                }
            }catch{
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
            }
        } while($cnt -lt $Max)
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $Max){
           Throw "could not read " + $fpath
        }
        #
        return $Q
        #
    }
    #
    <# -----------------------------------------
      Append-File
      append to the end of a file with error checking
      -------------------------------------------
     Usage:
     Open-CSVFile -fpath
        -fpath: file path to read in
     -------------------------------------------#>
    PopFile([string] $fpath = '',[Object] $fstring){
        #
        $cnt = 0
        $Max = 120
        $mxtstring = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
        #
        if (!(test-path $fpath)){
            New-Item -path $fpath -itemtype file -Force
        }
        #
        do{
            $mxtx = New-Object System.Threading.Mutex($false, $mxtstring)
            try{
                $imxtx = $mxtx.WaitOne(60 * 10)
                if($imxtx){
                    Add-Content -Path $fpath -Value $fstring -NoNewline -EA Stop
                    $mxtx.releasemutex()
                    break
                } else {
                    $cnt = $cnt + 1
                    Start-Sleep -s 3
                }
            }catch{
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
            }
        } while($cnt -lt $Max)
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $Max){
            Throw "could not write to " + $fpath
        }
        #
    }
     <# -----------------------------------------
      Import-CohortsInfo
      open the cohort info for the astropath
      processing pipeline with error checking from 
      the AstropathCohortsProgress.csv and 
      AstropathPaths.csv files in the mpath location
      -------------------------------------------
     Usage:
     Import-CohortsInfo -mpath 
     -mpath: main path for the astropath processing
       which contains all necessary processing files
     -------------------------------------------#>
    [PSCustomObject]ImportCohortsInfo([string] $mpath = ''){
        #
        $cohort_csv_file = $mpath + '\AstropathCohortsProgress.csv'
        #
        $project_data = $this.OpencsvFile($cohort_csv_file)
        #
        $paths_csv_file = $mpath + '\AstropathPaths.csv'
        #
        $paths_data = $this.opencsvfile($paths_csv_file)
        #
        $project_data = $this.MergeCustomObject( $project_data, $paths_data, 'Project')
        #
        return $project_data
        #
    }
    <# -------------------------------------------
     Import-SlideIDs
     open the AstropathAPIDdef.csv to get all slide
     available for processing

     Usage:
     Import-SlideIDs -mpath 
     -mpath: main path for the astropath processing
       which contains all necessary processing files
    # -------------------------------------------#>
    [PSCustomObject]ImportSlideIDs([string] $mpath = ''){
         #
         $defpath = $mpath + '\AstropathAPIDdef.csv'
         #
         $slide_ids = $this.opencsvfile( $defpath)
         return $slide_ids
        #
     }
    <# -------------------------------------------
     Merge-PSCustomObject
     Merge two PS Custom Objects based on a property
     return the left-outer-join in a new PS Custom Object

     Usage:
     Merge-PSCustomObject -d1 -d2 -property
    # -------------------------------------------#>
    [PSCustomObject]MergeCustomObject([PSCustomObject]$d1,
              [PSCustomObject]$d2, 
              [string]$property = ''
              ){
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
    }
    <# -------------------------------------------
     Convert-ObjectHash
     Convert a PSCutomObject to a Hash table

     Usage:
     Convert-ObjectHash -object [-columns]
     -object: object to convert
     [-columns]: columns of the object to use
     [-hash]: optional hash table to add to
    # -------------------------------------------#>
    [hashtable]ConvertObjectHash([PSCustomObject] $object,[Object] $columns = ''){
        #
        if (!($PSBoundParameters.ContainsKey('columns'))){
            $columns = ($object | Get-Member -MemberType NoteProperty).Name
        }
        #
        $hash = @{}
        #
        foreach($p in $columns){
            $hash.Add($p, $object.$p)
        }
        #
        return $hash
    }
    #
    [hashtable]ConvertObjectHash([PSCustomObject] $object,[Object] $columns = '',[Hashtable] $hash){
        #
        if (!($PSBoundParameters.ContainsKey('columns'))){
            $columns = ($object | Get-Member -MemberType NoteProperty).Name
        }
        #
        if (!($PSBoundParameters.ContainsKey('hash'))){
            $hash = @{}
        }
        #
        foreach($p in $columns){
            $hash.Add($p, $object.$p)
        }
        #
        return $hash
    }
}