<# -------------------------------------------
 sharedtools
 created by: Benjamin Green - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 general functions which may be needed by
 throughout the pipeline
 -------------------------------------------#>
 Class sharedtools{
    [string]$module
    [string]$mpath
    [string]$slideid
    [string]$psroot = $pshome + "\powershell.exe"
    [string]$coderoot
    [string]$pyinstalllocation = '\\'+$env:ComputerName+'\c$\users\public\astropath\py\'
    [string]$pyenv = $this.pyinstalllocation + 'astropathworkflow'
    [string]$pyinstalllog = $this.pyinstalllocation + 'pyinstall.log'
    <# -----------------------------------------
     OpenCSVFile
     open a csv file with error checking into a
     pscustom object
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: OpenCSVFile(fpath)
    ----------------------------------------- #>
    sharedtools(){}
    #
    [PSCustomObject]OpenCSVFile([string] $fpath){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $Max = 120
        $mxtxid = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
        
        #
        if (!(test-path $fpath)){
            New-Item -path $fpath -itemtype file -Force
        }
        #
        $Q = New-Object -TypeName psobject
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
            try{
                $Q = Import-CSV $fpath -ErrorAction Stop
                $e = 0
            }catch{
                $err = $_.Exception.Message
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
            }
            $this.ReleaseMxtx($mxtx, $fpath)
            #
        } while(($cnt -lt $Max) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $Max){
           Throw $cnt + ' attempts failed reading ' + $fpath + '. Final message: ' + $err
        }
        #
        return $Q
        #
    }
    <# -----------------------------------------
     GetContent
     open a file with error checking where each
     row is in a separate line
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: GetContent(fpath)
    ----------------------------------------- #>
    [Array]GetContent([string] $fpath){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $Max = 120
        $mxtxid = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
        
        #
        if (!(test-path $fpath)){
            New-Item -path $fpath -itemtype file -Force
        }
        #
        $Q = New-Object -TypeName psobject
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
            try{
                $Q = Get-Content $fpath -ErrorAction Stop
                $e = 0
            }catch{
                $err = $_.Exception.Message
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
            }
            $this.ReleaseMxtx($mxtx, $fpath)
            #
        } while(($cnt -lt $Max) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $Max){
           Throw $cnt + ' attempts failed reading ' + $fpath + '. Final message: ' + $err
        }
        #
        return $Q
        #
    }
    <# -----------------------------------------
     PopFile
     append to the end of a file
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: PopFile(fpath)
    ----------------------------------------- #>
    [void]PopFile([string] $fpath = '',[Object] $fstring){
        #
        $this.HandleWriteFile($fpath, $fstring, 'Pop')
        #
    }
    <# -----------------------------------------
     SetFile
     Overwrite a file
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: SetFile(fpath)
    ----------------------------------------- #>
    [void]SetFile([string] $fpath = '',[string] $fstring){
        #
        $this.HandleWriteFile($fpath, $fstring, 'Set')
        #
    }
    <# -----------------------------------------
     HandleReadFile
     write to a file with error checking
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: SetFile(fpath)
    ----------------------------------------- #>
    [void]HandleWriteFile([string] $fpath = '',[string] $fstring, [string] $opt){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $Max = 120
        $mxtxid = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
        #
        if (!(test-path $fpath)){
            New-Item -path $fpath -itemtype file -Force
        }
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
             try{
                $this.WriteFile($fpath, $fstring, $opt)
                $e = 0
             }catch{
                $err = $_.Exception.Message
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
             }
            $this.ReleaseMxtx($mxtx, $fpath)
            #
        } while(($cnt -lt $Max) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $Max){
            Throw $cnt + ' attempts failed writing ' + $fstring + ' to ' + $fpath + '. Final message: ' + $err
        }
        #
    }
    <# -----------------------------------------
     ReadFile
     append or overwrite a file
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: SetFile(fpath)
    ----------------------------------------- #>
     [void]WriteFile([string] $fpath = '',[string] $fstring, [string] $opt){
        if ($opt -eq 'Set'){
            Set-Content -Path $fpath -Value $fstring -NoNewline -EA Stop
        } elseif ($opt -eq 'Pop') {
            Add-Content -Path $fpath -Value $fstring -NoNewline -EA Stop
        }
     }
    <# -----------------------------------------
     GrabMxtx
     Grab my mutex, from: 
     https://stackoverflow.com/questions/7664490/interactively-using-mutexes-et-al-in-powershell
     ------------------------------------------
     Input: 
        -mxtxid: string object for a mutex
     ------------------------------------------
     Usage: GrabMxtx(mxtxid)
    ----------------------------------------- #>
    [System.Threading.Mutex]GrabMxtx([string] $mxtxid){
         try
            {
                $mxtx = New-Object System.Threading.Mutex -ArgumentList 'false', $mxtxid
                while (-not $mxtx.WaitOne(1000))
                {
                    Start-Sleep -m 500;
                }
                return $mxtx
            } 
            catch [System.Threading.AbandonedMutexException] 
            {
                $mxtx = New-Object System.Threading.Mutex -ArgumentList 'false', $mxtxid
                return $this.GrabMutex($mxtxid)
            }
    }
    <# -----------------------------------------
     ReleaseMxtx
     release mutex
     ------------------------------------------
     Input: 
        -mxtxid: string object for a mutex
     ------------------------------------------
     Usage: ReleaseMxtx(mxtx)
    ----------------------------------------- #>
    [void]ReleaseMxtx([System.Threading.Mutex]$mxtx, [string] $fpath){
        try{
            $mxtx.ReleaseMutex()
            try { $mxtx.ReleaseMutex() } catch {} # if another process crashes the mutex is never given up.
        } catch {
            Throw "mutex not released: " + $fpath
        }
    }
    <# -----------------------------------------
     ImportCohortsInfo
     open the cohort info for the astropath
     processing pipeline with error checking from 
     the AstropathCohortsProgress.csv and 
     AstropathPaths.csv files in the mpath location
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportCohortsInfo(mpath)
    ----------------------------------------- #>
    [PSCustomObject]ImportCohortsInfo([string] $mpath){
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
    <# -----------------------------------------
     ImportConfigInfo
     open the config info for the astropath
     processing pipeline with error checking from 
     the AstropathConfig.csv in the mpath location
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportCohortsInfo(mpath)
    ----------------------------------------- #>
    [PSCustomObject]ImportConfigInfo([string] $mpath){
        #
        $config_csv_file = $mpath + '\AstropathConfig.csv'
        #
        $config_data = $this.opencsvfile($config_csv_file)
        #
        return $config_data
        #
    }
    <# -----------------------------------------
     ImportSlideIDs
     open the AstropathAPIDdef.csv to get all slide
     available for processing
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportSlideIDs(mpath)
    ----------------------------------------- #>
    [PSCustomObject]ImportSlideIDs([string] $mpath){
        #
        $defpath = $mpath + '\AstropathAPIDdef.csv'
        #
        $slide_ids = $this.opencsvfile( $defpath)
        return $slide_ids
        #
     }
    <# -----------------------------------------
     ImportFlatfieldModels
     open the AstropathAPIDdef.csv to get all slide
     available for processing
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportFlatfieldModels(mpath)
    ----------------------------------------- #>
    [PSCustomObject]ImportFlatfieldModels([string] $mpath){
        #
        $defpath = $mpath + '\AstroPathFlatfieldModels.csv'
        #
        $slide_ids = $this.opencsvfile( $defpath)
        return $slide_ids
        #
     }
    <# -----------------------------------------
     MergePSCustomObject
     Merge two PS Custom Objects based on a property
     return the left-outer-join in a new PS Custom Object
     ------------------------------------------
     Usage: MergePSCustomObject(d1, d2, property)
    ----------------------------------------- #>
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
    [void]defCodeRoot(){
        #
        if ($PSScriptRoot[0] -ne '\'){
            $root = ('\\' + $env:computername+'\'+$PSScriptRoot) -replace ":", "$"
        } else{
            $root = $PSScriptRoot -replace ":", "$"
        }
        #
        $folder = $root -Split('\\astropath\\')
        $this.coderoot = $folder[0] + '\\astropath'
    }
    #
    [string]GetVersion($mpath, $module, $project){
        #
        $configfile = $this.ImportConfigInfo($mpath)
        $vers = ($configfile | Where-Object {$_.Project -eq $project}).($module+'version')
        if (!$vers){
            Throw 'No version number found'
        } elseif ($vers -ne '0.0.1'){
            $this.checkconda()
            $this.checkpyapenvir()
        } elseif ($module -contains  @('meanimagecomparison', 'warping')){
            Throw 'module not supported in this version (' + $vers + '): ' + $module
        }
        return $vers
        #
    }
    <# -----------------------------------------
     checkconda
     Check if conda is a command in Powershell 
     if not add it to the environment path as a
     command
     ------------------------------------------
     Usage: $this.checkconda()
    ----------------------------------------- #>
    [void]CheckConda(){
        #
        # check if conda is a
        #
        if (!(test-path C:\ProgramData\Miniconda3)){
            Throw "Miniconda must be installed for this code version"
        }
        #
        try{
            conda *>> NULL
        }catch{
            if($_.Exception.Message -match "The term 'conda' is not"){
                    #
                    $env:PATH = $env:PATH + "C:\ProgramData\Miniconda3;C:\ProgramData\Miniconda3\Library\mingw-w64\bin;
                        C:\ProgramData\Miniconda3\Library\usr\bin;C:\ProgramData\Miniconda3\Library\bin;
                        C:\ProgramData\Miniconda3\Scripts;C:\ProgramData\Miniconda3\bin;"
                    #
                    (& "C:\ProgramData\Miniconda3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
                    #
                }
             }
        }
     <# -----------------------------------------
     astropathpypath
     return the path to the astropath python 
     package root location.
     ------------------------------------------
     Usage: $this.astropathpypath()
    ----------------------------------------- #>    
    [string]AstroPathpyPath(){
        $this.defCodeRoot()
        $astropathpypath = $this.coderoot + '\..\.'
        return($astropathpypath)
    }
     <# -----------------------------------------
     CheckpyAPEnvir
     Check if py\astropathworkflow conda environment
     exists. If it does not create it and install
     the astropath package from the current working.
     If it does exist check for updates.
     ------------------------------------------
     Usage: $this.CheckpyAPEnvir()
    ----------------------------------------- #>        
    [void]CheckpyAPEnvir(){
        #
        $this.createdirs($this.pyinstalllocation)
        $envs = conda info --envs
        if(!($envs -match $this.pyenv)){
            $this.createpyapenvir()
        }
        #
    }
    #
    [void]CheckpyAPEnvir($u){
        #
        $this.createdirs($this.pyinstalllocation)
        $envs = conda info --envs
        if($envs -match $this.pyenv){
            $this.upgradepyapenvir()
        } else {
            $this.createpyapenvir()
        }
        #
    }
    <# -----------------------------------------
     CreatepyAPEnvir
     create py\astropathworkflow conda environment.
     ------------------------------------------
     Usage: $this.createpyapenvir()
    ----------------------------------------- #>      
    [void]CreatepyAPEnvir(){
        conda create -y -p $this.pyenv python=3.8 2>&1 >> $this.pyinstalllog
        $this.pyenv + " CONDA ENVIR CREATED" | Out-File $this.pyinstalllog
        conda activate $this.pyenv 2>&1 >> $this.pyinstalllog
        $this.pyenv + " CONDA ENVIR ACTIVATED" | Out-File $this.pyinstalllog
        conda install -y -c conda-forge pyopencl gdal cvxpy numba 'ecos!=2.0.8' git 2>&1 >> $this.pyinstalllog
        $this.pyenv + " CONDA ENVIR INSTALLS COMPLETE" | Out-File $this.pyinstalllog
        pip -q install $this.astropathpypath() 2>&1 >> $this.pyinstalllog
        $this.pyenv + " PIP INSTALLS COMPLETE" | Out-File $this.pyinstalllog
        conda deactivate $this.pyenv 2>&1 >> $this.pyinstalllog
    }
    <# -----------------------------------------
     upgradepyapenvir
     create py\astropathworkflow conda environment.
     ------------------------------------------
     Usage: $this.upgradepyapenvir()
    ----------------------------------------- #>    
    [void]UpgradepyAPEnvir(){
        conda activate $this.pyenv 2>&1 >> $this.pyinstalllog
        pip -q install -U $this.astropathpypath()  2>&1 >> $this.pyinstalllog
        conda deactivate $this.pyenv 2>&1 >> $this.pyinstalllog
    }
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
}