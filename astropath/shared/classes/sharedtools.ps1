<# -------------------------------------------
 sharedtools
 created by: Benjamin Green - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 general functions which may be needed by
 throughout the pipeline
 -------------------------------------------#>
 Class sharedtools : aptabletools {
    [string]$module
    [string]$mpath
    [string]$slideid
    [string]$psroot = $pshome + "\powershell.exe"
    [string]$package = 'astropath'
    [array]$modules
    [hashtable]$module_project_data = @{}
    [hashtable]$modulelogs = @{}
    [switch]$checkpyenvswitch = $false
    [switch]$teststatus = $false
    [array]$newtasks
    [string]$processname
    [string]$processid
    [hashtable]$softwareurls = @{
        'Miniconda3' = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe';
        'MikTeX' = '';
        'matlab' = '';
        'git' = '';
        'python' = '';
        'NET' = ''
    }
    [hashtable]$softwareargs = @{
        'Miniconda3' = @('/InstallationType=AllUsers', '/RegisterPython=1','/S','/D=C:\ProgramData\Miniconda3')
        'MikTeX' = @('');
        'matlab' = @('');
        'git' = @('');
        'python' = @('');
        'NET' = @('')
    }
    #
    sharedtools(){
        $this.processname = ([System.Diagnostics.Process]::GetCurrentProcess()).name 
        $this.processid = ([System.Diagnostics.Process]::GetCurrentProcess()).ID 
    }
    #
    [string]pyinstalllocation(){
         $str = '\\' + $this. defserver() + 
                '\c$\users\public\' + $this.package +'\py\'
        return $str
    }
    #
    [string]pyenv(){
        #
        if ($this.teststatus){
            $str = $this.pyinstalllocation() + $this.package + 'workflow-test'
        } else {
            $str = $this.pyinstalllocation() + $this.package + 'workflow'
        }
        #
        return $str
        #
    }
    #
    [string]pyinstalllog(){
        $str = $this.pyinstalllocation() + 'pyinstall.log'
        return $str
    }
    #
    [string]pypackagepath(){
        $str = $this.coderoot() + '\..\.'
        if (!$this.isWindows()){
            $str = $str -replace ('\\', '/')
        }
        #
        return $str
    }
    #
    [string]testpackagepath(){
        $str = $this.coderoot() + '\..\test\data'
        if (!$this.isWindows()){
            $str = $str -replace ('\\', '/')
        }
        #
        return $str
    }
    <# -----------------------------------------
     CodeRoot
     the path to the powershell module (package)
     ------------------------------------------
     Usage: $this.CodeRoot()
    ----------------------------------------- #>
    [string]coderoot(){
        #
        $root = $this.defRoot()
        $folder = $root -Split('\\' + $this.package + '\\')
        $str = $folder[0] + '\' + $this.package
        return $str
        #
    }
    <# -----------------------------------------
     GetVersion
     get the version number and check that the 
     module is supported by that version. If 
     the version number was added as >v0.0.1
     then the version number should be selected
     based on the full git development tag. If
     the version number was added as v0.0.1
     just return v0.0.1.
     ------------------------------------------
     Usage: $this.GetVersion()
    ----------------------------------------- #>
    [string]GetVersion($mpath, $module, $project){
        #
        $this.ImportCohortsInfo($mpath)
        $projectconfig = $this.full_project_dat | 
            & { process { if ($_.Project -eq $project) { $_ }}}
        if (!$projectconfig){
            Throw ('Project not found for project number: '  + $project)
        }    
        #
        $vers = $projectconfig.($module+'version')    
        if (!$vers){
            $this.checkmoduleexists($mpath, $module)
        }
        #
        Write-Host 'THIS IS HAPPENING'
        if ($this.apversionchecks($mpath, $module, $vers)){
            return ("v" + $vers)
        } else {
            #$vers = $this.getfullversion()
            #$this.upgradepyenvir()
            $this.checkconda()
            $vers = $this.getversionpy()
            Write-Host 'Getting this version:' $vers
        }
        # 
        return $vers
        #
    }
    #
    [string]GetVersion($mpath, $module, $project, $short){
        #
        $this.ImportCohortsInfo($mpath) 
        #
        $projectconfig = $this.full_project_dat | 
            & { process { if ($_.Project -eq $project) {$_}}}
        if (!$projectconfig){
            Throw ('Project not found for project number: '  + $project)
        }    
        #
        $vers = $projectconfig.($module+'version')    
        if (!$vers){
           $this.checkmoduleexists($mpath, $module)
           $vers = '0.0.2'
        }
        #
        return ("v" + $vers)
        #
    }
    <# -----------------------------------------
     APVersionChecks
     version checking applied specifically for the 
     astropath pipeline. This is the only method
     specific to the astropath package.
     ------------------------------------------
     Usage: $this.APVersionChecks()
    ----------------------------------------- #>
    [switch]APVersionChecks($mpath, $module, $vers){
        #
        if ($this.package -match 'astropath' -and
            $vers -match '0.0.1'){
            return $true
        } else {
            return $false
        }
        #
    }
    [void]checkmoduleexists($mpath, $module){
        $this.importdependencyinfo($mpath) 
        $headers = '^' + ($this.dependency_data.module -join '$|^') + '$'
           if ($module -notmatch $headers){
               write-host 'Valid module regex:' $headers
                throw ($module + 
                    ' not defined. Please define module '+
                    'in AstroPathDependency and optionally AstroPathConfig for version selection'
                )
            }
    }
    <# -----------------------------------------
    getfullversion
    get the full version number for this branch.
    if the path is a git repository pull the 
    github version and check if the git branch
    is clean. If it isn't add the date. If the
    path is not a git repo try to get the version
    number from the conda envrionment, if the
    package \ envir do not exist return the 
    v0.0.0... version number.
    This logic is from:
    https://pypi.org/project/setuptools-scm/ 
    ------------------------------------------
     Usage: $this.getfullversion()
    ----------------------------------------- #>
    [string]getfullversion(){
        #
        if ($this.checkgitrepo()){
            $version = $this.getgitversion()
            if (!($this.checkgitstatus())){
                $version = $version, $this.getdate() -join '.'
            } 
        } else {
            $version = $this.getpackageversion() 
            $version = $version
        }
        #
        return $version
    }
    <# -----------------------------------------
    getdate
    ------------------------------------------
     Usage: $this.getdate()
     get the date in the version formatting
    ----------------------------------------- #>
    [string]getdate(){
        return ("d"+(Get-Date -format "yyyyMMdd"))
    }
    <# -----------------------------------------
    checkgitinstalled
    ------------------------------------------
     Usage: $this.checkgitinstalled()
     if git is a command then returns true 
     else returns false
    ----------------------------------------- #>
    [switch]checkgitinstalled(){
        try {
            git --version | Out-NUll
            return $true
        } catch {
            return $false
        }
    }
    <# -----------------------------------------
    checkgitrepo
    if git is installed check if the tree is a
    git repo and return true. If git is not 
    installed or it is not a git repo will return
    false
    ------------------------------------------
     Usage: $this.checkgitrepo()
    ----------------------------------------- #>
    [switch]checkgitrepo(){
        if ($this.checkgitinstalled()){
            try {
               git -C $this.pypackagepath() rev-parse --is-inside-work-tree | Out-Null
               return $true
            } catch {
               return $false
            }
        } else {
            return $false
        }
    }
    <# -----------------------------------------
    checkgitstatus
    ------------------------------------------
     Usage: $this.checkgitstatus()
     check the git status, if the working tree is
     clean return true else return false
    ----------------------------------------- #>
    [switch]checkgitstatus(){
           $gitstatus = git -C $this.pypackagepath() status
           if ($gitstatus -match "nothing to commit, working tree clean"){
                return $true
           } else {
                return $false
           }
    }
    #
    [switch]checkgitstatustest(){
        $gitstatus = git -C $this.testpackagepath() status
        if ($gitstatus -match "nothing to commit, working tree clean"){
             return $true
        } else {
             return $false
        }
    }
    [void]checkgitsubmodule($submodule){
        $gitsubmodule = git submodule status
        $modulematch = $gitsubmodule -match $submodule
        if ($null -ne $modulematch) {
            $linematch = '^-.*' + $submodule
            if ($modulematch -match $linematch) {
                git submodule update --init
            }
        }
    }
    <# -----------------------------------------
    getgitversion
    ------------------------------------------
     Usage: $this.checkgitstatus()
     get the git version in the astropath format
    ----------------------------------------- #>
    [string]getgitversion(){
        $v = git -C $this.pypackagepath() describe --tags --long
        $v2 = $v -split '-'
        $v3 = $v2[0] -split '\.'
        $v4 = [int]$v3[2] + 1
        $v5 = $v3[0], $v3[1], $v4, ('dev'+$v2[1] + '+' + $v2[2]) -join '.'
        return $v5
    }
    <# -----------------------------------------
    getpackageversion
    ------------------------------------------
     Usage: $this.getpackageversion()
     get the package version in the astropath 
     format from the enirvonment install. if the
     package or environment is not installed 
     for some reason return the v0.0.0 versioning
    ----------------------------------------- #>
    [string]getpackageversion(){
        $version = "v0.0.0.dev0+g0000000", $this.getdate() -join '.' 
        if ($this.CheckpyEnvir()){
            #
            $this.checkconda()
            $condalist = conda list -p $this.pyenv()
            $astropath = $condalist -match $this.package
            if ($astropath[1]){
                $version = 'v'+(($astropath[1] -split ' ') -match 'dev')
            }
        }
        return $version                
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
        if (!$this.isWindows()){
            return
        }
        #
        $server = $this.defServer()
        $drive = '\\'+$server+'\C$'
        #
        $minicondapath = ($drive + '\ProgramData\Miniconda3')
        $this.testcondainstall($minicondapath)
        #
        if ((get-module).name -notcontains 'Conda'){
            $Env:CONDA_EXE = $minicondapath, 'Scripts\conda.exe' -join '\'
            $Env:_CE_M = ""
            $Env:_CE_CONDA = ""
            $Env:_CONDA_ROOT = $minicondapath
            $Env:_CONDA_EXE =  $minicondapath, 'Scripts\conda.exe' -join '\'
            $CondaModuleArgs = @{ChangePs1 = $True}
            $mname = "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1"
            Import-Module $mname -Global
            #
        }
        #
        $this.testcondapython()
    }
    #
    [void]testcondainstall($minicondapath){
        #
        if (!(test-path $minicondapath )){
            $this.softwareinstall('Miniconda3', $minicondapath)
        }
        #
        if (!(test-path $minicondapath )){
            Throw "Miniconda must be installed for this code version " + 
                $minicondapath
        }
        #
    }
    #
    [void]testcondapython(){
        #
        $pyscript = $PSScriptRoot + '\..\condapython.py'
        conda activate $this.pyenv()
        try {
            python $pyscript
            conda deactivate
        }
        catch {
            conda deactivate
            Throw $_.Exception.Message
        }
    }
    #
    [String]getversionpy(){
        #
        $pyscript = $PSScriptRoot + '\..\versionpython.py'
        $version = ""
        conda activate $this.pyenv()
        try {
            $version = python $pyscript
            conda deactivate
        }
        catch {
            conda deactivate
            Throw $_.Exception.Message
        }
        return $version
    }
    <# ------------------------------------------
    CheckMikTex
    ------------------------------------------
    check if miktex is installed and deliver
    a warning to the console if it is not
    ------------------------------------------ #>
    [void]checkmitex(){
        $myenv = ($env:PATH -split ';')
        if ($myenv -notmatch 'MikTeX'){
            Write-Host "WARNING: MikTex command does not exist on the current system!"
        }
    }
    <# ------------------------------------------
    Checkmatlab
    ------------------------------------------
    check if matlab is installed and deliver
    a warning to the console if it is not
    ------------------------------------------ #>
    [void]checkmatlab(){
        $myenv = ($env:PATH -split ';')
        if ($myenv -notmatch 'matlab'){
            Write-Host "WARNING: matlab command does not exist on the current system! v0.0.1 software not supported w/o matlab."
        }        
    }
    #
    [void]checkNET(){
        if ($this.iswindows()){
            if (!(Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full").Release -ge 461808){
                Throw 'Please install .NET framework 4.7.2 or greater to run code'
            }
        }
    }
    #
    [void]checksoftware(){
        $this.checkNET()
    }
     <# -----------------------------------------
     CheckpyEnvir
     Check if py\<packagename>workflow conda environment
     exists. If it does not create it and install
     the astropath package from the current working.
     If it does exist check for updates. Added 
     a switch that assumes that the conda environment
     is active. Should check the conda environment
     higher up in the code. 
     ------------------------------------------
     Usage: $this.CheckpyEnvir()
    ----------------------------------------- #>        
    [switch]CheckpyEnvir(){
        #
        $this.checkconda()
            try {
                conda activate $this.pyenv() | Out-Null
                conda deactivate | Out-Null
                return $true
            } catch {
                return $false
            }
        #
    }
    #
    <# -----------------------------------------
     CreatepyEnvir
     create py\<packagename>workflow conda environment.
     ------------------------------------------
     Usage: $this.createpyenvir()
    ----------------------------------------- #>      
    [void]CreatepyEnvir(){
        $this.checkconda()
        $this.createdirs($this.pyinstalllocation())
        conda create -y -p $this.pyenv() python=3.8 2>&1 >> $this.pyinstalllog()
        $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $this.PopFile($this.pyinstalllog(), ($this.pyenv() + 
            " CONDA ENVIR CREATED; $time `r`n"))
        conda activate $this.pyenv() 2>&1 >> $this.pyinstalllog()
        $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $this.PopFile($this.pyinstalllog(), ($this.pyenv() + 
            " CONDA ENVIR ACTIVATED; $time  `r`n"))
        conda install -y -c conda-forge pyopencl gdal cvxpy numba 'ecos!=2.0.8' git `
              2>&1 >> $this.pyinstalllog()
        $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $this.PopFile($this.pyinstalllog(), ($this.pyenv() + 
            " CONDA ENVIR INSTALLS COMPLETE; $time  `r`n"))
        pip install $this.pypackagepath() 2>&1 >> $this.pyinstalllog()
        $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $this.PopFile($this.pyinstalllog(), ($this.pyenv() + 
            " PIP INSTALLS COMPLETE; $time `r`n"))
        conda deactivate 2>&1 >> $this.pyinstalllog()
    }
    <# -----------------------------------------
     upgradepyenvir
     create py\<packagename>workflow conda environment.
     ------------------------------------------
     Usage: $this.upgradepyenvir()
    ----------------------------------------- #>    
    [void]UpgradepyEnvir(){
        try{
            $this.checkconda()
            conda activate $this.pyenv() 2>&1 >> $this.pyinstalllog()
            git fetch --all --tags
            $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $this.PopFile($this.pyinstalllog(), ($this.pyenv() + 
                " CONDA ENVIR ACTIVATED; $time  `r`n"))
            pip install -U $this.pypackagepath() 2>&1 >> $this.pyinstalllog()
            $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $this.PopFile($this.pyinstalllog(), ($this.pyenv() + 
                " PIP INSTALLS COMPLETE; $time `r`n"))
            conda deactivate 2>&1 >> $this.pyinstalllog()
        } catch {
            $this.createpyenvir()
        }
    }
    #
    [void]softwareinstall($name){
        $this.createdirs($this.pyinstalllocation())
        $this.softwaredownload($name)
        $this.softwareexe($name)
        $this.removefile($this.softwarelinkpath($name))
    }
    #
    [void]softwareinstall($name, $path){
        Write-Verbose ('INSTALLING:' + $name)
        $this.createdirs($this.pyinstalllocation())
        $this.softwaredownload($name)
        $this.softwareexe($name, $path)
        $this.removefile($this.softwarelinkpath($name))
    }
    #
    [void]softwaredownload($name){
            $url = $this.softwareurls.($name)
            curl $url -o $this.softwarelinkpath($name)
    }
    #
    [void]softwareexe($name){
        Start-Process -Filepath $this.softwarelinkpath($name) `
            -ArgumentList $this.softwareargs.($name) -Wait -Verb RunAs
    }
    #
    [void]softwareexe($name, $path){
        $newargs = $this.softwareargs.($name) `
            -replace [regex]::escape('C:\ProgramData\' + $name ), $path
        Start-Process -Filepath $this.softwarelinkpath($name) `
            -ArgumentList $newargs -Wait -Verb RunAs
    }
    #
    [string]softwarelinkpath($name){
            return ($this.pyinstalllocation() + $name + ".exe")
    }
    <# -----------------------------------------
     progressindicator
     ------------------------------------------
     Usage: $this.progressindicator()
    ----------------------------------------- #>    
    [void]progressindicator($previous, $current, $c, $ctotal){
        #
        $prepend = 'Checking Slides'
        $append = '% Complete'
        $p = [math]::Round(100 * ($c / $ctotal))
        $writecurrent = "`r" + $prepend + ': ' + $p + $append + ' ' + $current
        Write-Host -NoNewline $($writecurrent)
        #
    }
    #
    [void]progressbar($c, $ctotal, $current){
        #
        $p = [math]::Round(100 * ($c / $ctotal))
        Write-Progress -Activity "Checking slides" `
            -Status ("$p% Complete: Slide " +  $current)`
            -PercentComplete $p `
            -CurrentOperation $current
        #
    }
    #
    [void]progressbarfinish(){
        write-progress -Activity "Checking slides" -Status "100% Complete:" -Completed
    }
    #
    <#
        reads in the module names from the dependency data.
        Optional overload (1) will force update the modulenames.
        Otherwise this just checks if the modules were previously
        defined
    #>
    [void]getmodulenames(){
        #
        if (!$this.modules){
            $this.getmodulenames($true)
        }
        #
    }
    #
    [void]getmodulenames($update){
        #
        $this.ImportDependencyInfo() 
        $this.modules = $this.dependency_data.module 
        #
    }
    <#
        updates the module status for each module.
        optional overload (1) to update for a specified 
        module
    #>
    [void]updatemodulestatus(){
        #
        $this.getmodulenames()
        $this.modules | & { process {
            $this.updatemodulestatus($_)
        }}
        #
    }
    #
    [void]updatemodulestatus($module){
        #
        $this.getmodulestatus($module, $true)
        #
    }
    <#
        defines $this.module_project_data hashtable for each
        module, i.e. a list of active projects for the module
        referenced by $this.module_project_data.($module). Optional 
        overload (1) for a specified module and (2) for a 
        specfied module and to force update. If projects
        to use are specified it will use those.
    #>
    [void]getmodulestatus(){
        #
        $this.getmodulenames()
        $this.modules | & { process {
            $this.getmodulestatus($_)
        }}
        #
    }
    #
    [void]getmodulestatus($module){
        #
        if (!$this.module_project_data.($module)){
            $this.getmodulestatus($module, $true)
        }
        #
    }
    #
    [void]getmodulestatus($module, $update){
        #
        if (!$this.projects){
            $this.module_project_data.($module) = $this.GetAPProjects($module)
        } else {
            $this.module_project_data.($module) = $this.projects
       }
        #
    }
     #
     [void]getmodulelogs(){
        $this.getmodulelogs($false)
    }
    #
    [void]getmodulelogs($createwatcher){
        #
        $this.getmodulenames()
        foreach ($cmodule in $this.modules){
            if ($this.modulelogs.($cmodule).count -eq 0){
                $this.modulelogs.($cmodule) = @{} 
            }
            $this.allprojects | &{ process {
                #
                $this.getmodulelogs($cmodule, $_, $createwatcher )
                #
            }}
            #
        }
        #
    }
    [void]getmodulelogs($cmodule, $project){
        $this.getmodulelogs($cmodule, $project, $false)
    }
    #
    [void]getmodulelogs($cmodule, $project, $createwatcher){
        #
        $newlog = $this.importlogfile($cmodule, $project, $createwatcher)
        $this.getnewloglines($newlog, $project, $cmodule)
        $this.modulelogs.($cmodule).($project) = $newlog
        #
    }
    #
    [void]getnewloglines($newlog, $project, $cmodule){
        #
        $newlog_finishlines =  $newlog | 
            & { process { 
                if(
                    $_.Message -match ('^' + $this.log_finish + '|^' + $this.log_start)
                ) { $_ }
            }}
        #
        if ($newlog_finishlines){
            #
            if ($this.modulelogs.($cmodule).($project)){
                $slidestocheck = (
                    compare-object $newlog_finishlines $this.modulelogs.($cmodule).($project) `
                        -Property 'SlideID','Date' |
                    & { process { 
                        if ($_.SideIndicator -eq '<=') { $_ }
                    }}
                ).SlideID
            } else {
                $slidestocheck = $newlog_finishlines.slideid
            } 
            #
            if ($cmodule -match 'batch'){
                $slidestocheck = $slidestocheck |
                 &{ process {
                    $this.getbatchslideslight($_, $project)
                }}
            }
            #
            $this.addnewtasks($slidestocheck)
            #
        }
    }
    #
    [void]addnewtasks($slidestocheck){
        #
        if (!$this.newtasks){
            $this.newtasks = @()
        }
        #
        $this.newtasks += $slidestocheck
        $this.newtasks = ($this.newtasks | Sort-Object | Get-Unique)
        #
    }   
    #
    [array]getbatchslideslight([string]$mbatchid, $cproject){
        #
        if ($mbatchid[0] -match '0'){
            [string]$mbatchid = $mbatchid[1]
        }
        #
        $batch = $this.slide_data | & { process { 
            if (
                $_.BatchID -eq $mbatchid.trim() -and 
                $_.Project -eq $cproject.trim()
            ) {$_}}}
        #
        if ($batch){
            return $batch.SlideID
        }
        #
        return @()
        #
    }
    #  
}