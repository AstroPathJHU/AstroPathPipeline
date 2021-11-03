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
    #
    sharedtools(){}
    #
    [string]pyinstalllocation(){
         $str = '\\' + $this. defserver() + 
                '\c$\users\public\' + $this.package +'\py\'
        return $str
    }
    #
    [string]pyenv(){
        $str = $this.pyinstalllocation() + $this.package + 'workflow'
        return $str
    }
    #
    [string]pyinstalllog(){
        $str = $this.pyinstalllocation() + 'pyinstall.log'
        return $str
    }
    #
    [string]pypackagepath(){
        $str = $this.coderoot() + '\..\.'
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
        $configfile = $this.ImportConfigInfo($mpath)
        $vers = ($configfile | 
            Where-Object {$_.Project -eq $project}).($module+'version')
        if (!$vers){
            Throw 'No version number found'
        }
        #
        if ($this.apversionchecks($mpath, $module, $vers)){
            return ("v" + $vers)
        }
        # 
        $vers = $this.getfullversion()
        return $vers
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
        if (
            ($module -contains  @('meanimagecomparison', 'warping')) -and 
                $vers -eq '0.0.1'
            ){
            #
            Throw 'module not supported in this version (' + $vers + 
                '): ' + $module
            #
        } elseif (
            $module -contains  @('batchflatfield') -and 
                $vers -ne '0.0.1' 
            ) {
            #
            Throw 'batchflatfield is run from the meanimagecomparison module ' +
                'and is not initiated in powershell for version: ' + $vers    
            #
        }
        #
        if ($this.package -match 'astropath' -and $vers -eq '0.0.1'){
            return $true
        } else {
            return $false
        }
        #
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
            $version = $version, $this.getdate() -join '.' 
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
            $gitversion = git --version
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
               $gitrepo = git -C $this.pypackagepath rev-parse --is-inside-work-tree
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
        $version = "v0.0.0.dev0+g0000000"
        if ($this.CheckpyEnvir()){
            #
            $this.checkconda()
            $condalist = conda list -p $this.pyenv()
            $astropath = $condalist -match $this.package
            if ($astropath[1]){
                $version = 'v'+($astropath[1] -split ' ') -match 'dev'
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
        $server = $this.defServer()
        $drive = '\\'+$server+'\C$'
        #
        $minicondapath = ($drive + '\ProgramData\Miniconda3')
        if (!(test-path $minicondapath )){
            Throw "Miniconda must be installed for this code version " + 
                $minicondapath
        }
        #
        try{
            conda *>> NULL
        }catch{
            if($_.Exception.Message -match "The term 'conda' is not"){
                    #
                    $str = ";C:\ProgramData\Miniconda3;" +
                           "C:\ProgramData\Miniconda3\Library\mingw-w64\bin;" +
                           "C:\ProgramData\Miniconda3\Library\usr\bin;" +
                           "C:\ProgramData\Miniconda3\Library\bin;" +
                           "C:\ProgramData\Miniconda3\Scripts;"+
                           "C:\ProgramData\Miniconda3\bin;"
                    $env:PATH += $str.replace('C:', $drive)
                    #
                    $str = ("C:\ProgramData\Miniconda3\Scripts\conda.exe").replace('C:', $drive)
                    (& $str "shell.powershell" "hook") | Out-String | Invoke-Expression
                    #
                }
             }
        }
     <# -----------------------------------------
     CheckpyEnvir
     Check if py\<packagename>workflow conda environment
     exists. If it does not create it and install
     the astropath package from the current working.
     If it does exist check for updates.
     ------------------------------------------
     Usage: $this.CheckpyEnvir()
    ----------------------------------------- #>        
    [switch]CheckpyEnvir(){
        #
        $this.checkconda()
        try {
            conda activate $this.pyenv() 2>&1 >> $this.pyinstalllog()
            conda deactivate $this.pyenv() 2>&1 >> $this.pyinstalllog()
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
        $this.PopFile($this.pyinstalllog(), ($this.pyenv() + " CONDA ENVIR CREATED"))
        conda activate $this.pyenv() 2>&1 >> $this.pyinstalllog()
        $this.PopFile($this.pyinstalllog(), ($this.pyenv() + " CONDA ENVIR ACTIVATED"))
        conda install -y -c conda-forge pyopencl gdal cvxpy numba 'ecos!=2.0.8' git `
              2>&1 >> $this.pyinstalllog()
        $this.PopFile($this.pyinstalllog(), ($this.pyenv() + " CONDA ENVIR INSTALLS COMPLETE"))
        pip -q install $this.pypackagepath 2>&1 >> $this.pyinstalllog()
        $this.PopFile($this.pyinstalllog(), ($this.pyenv() + " PIP INSTALLS COMPLETE"))
        conda deactivate $this.pyenv() 2>&1 >> $this.pyinstalllog()
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
            pip -q install -U $this.pypackagepath()  2>&1 >> $this.pyinstalllog()
            conda deactivate $this.pyenv() 2>&1 >> $this.pyinstalllog()
        } catch {
            $this.createpyenvir()
        }
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
}