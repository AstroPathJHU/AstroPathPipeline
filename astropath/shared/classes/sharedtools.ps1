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
    [string]$coderoot
    [string]$pyinstalllocation
    [string]$pyenv 
    [string]$pyinstalllog
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
    sharedtools(){
        $this.createpypaths()
    }
    #
    [void]createpypaths(){
        #
        $this.pyinstalllocation = '\\'+$this.defserver()+'\c$\users\public\astropath\py\'
        $this.pyenv = $this.pyinstalllocation + 'astropathworkflow'
        $this.pyinstalllog = $this.pyinstalllocation + 'pyinstall.log'
        #
    } 
    #
    [void]defCodeRoot(){
        #
        $root = $this.defRoot()
        #
        $folder = $root -Split('\\astropath\\')
        $this.coderoot = $folder[0] + '\astropath'
    }
    #
    [string]GetVersion($mpath, $module, $project){
        #
        $configfile = $this.ImportConfigInfo($mpath)
        $vers = ($configfile | Where-Object {$_.Project -eq $project}).($module+'version')
        if (!$vers){
            Throw 'No version number found'
        } elseif ($vers -ne '0.0.1'){
            if ($module -contains  @('batchflatfield')){
                Throw 'batchflatfield is run from the meanimagecomparison module ' +
                    'and is not initiated in powershell for version: ' + $vers    
            }
            #
            try {
                $l = $this.getpythonvers()
            } catch{
                $this.checkconda()
                $this.checkpyapenvir()
                $l = $this.getpythonvers()
            }
            #
            if ($vers -match $l){
                $vers = $l
            }
            #
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
        $server = $this.defServer()
        $drive = '\\'+$server+'\C$'
        #
        $minicondapath = ($drive + '\ProgramData\Miniconda3')
        if (!(test-path $minicondapath )){
            Throw "Miniconda must be installed for this code version " + $minicondapath
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
        try {
            conda activate $this.pyenv 2>&1 >> $this.pyinstalllog
            conda deactivate $this.pyenv 2>&1 >> $this.pyinstalllog
        } catch {
            $this.createpyapenvir()
        }
        #
    }
    #
    [void]CheckpyAPEnvir($u){
        #
        $this.upgradepyapenvir()
        #
    }
    <# -----------------------------------------
     CreatepyAPEnvir
     create py\astropathworkflow conda environment.
     ------------------------------------------
     Usage: $this.createpyapenvir()
    ----------------------------------------- #>      
    [void]CreatepyAPEnvir(){
        $this.createdirs($this.pyinstalllocation)
        conda create -y -p $this.pyenv python=3.8 2>&1 >> $this.pyinstalllog
        $this.PopFile($this.pyinstalllog, ($this.pyenv + " CONDA ENVIR CREATED"))
        conda activate $this.pyenv 2>&1 >> $this.pyinstalllog
        $this.PopFile($this.pyinstalllog, ($this.pyenv + " CONDA ENVIR ACTIVATED"))
        conda install -y -c conda-forge pyopencl gdal cvxpy numba 'ecos!=2.0.8' git `
              2>&1 >> $this.pyinstalllog
        $this.PopFile($this.pyinstalllog, ($this.pyenv + " CONDA ENVIR INSTALLS COMPLETE"))
        pip -q install $this.astropathpypath() 2>&1 >> $this.pyinstalllog
        $this.PopFile($this.pyinstalllog, ($this.pyenv + " PIP INSTALLS COMPLETE"))
        conda deactivate $this.pyenv 2>&1 >> $this.pyinstalllog
    }
    <# -----------------------------------------
     upgradepyapenvir
     create py\astropathworkflow conda environment.
     ------------------------------------------
     Usage: $this.upgradepyapenvir()
    ----------------------------------------- #>    
    [void]UpgradepyAPEnvir(){
        try{
            conda activate $this.pyenv 2>&1 >> $this.pyinstalllog
            pip -q install -U $this.astropathpypath()  2>&1 >> $this.pyinstalllog
            conda deactivate $this.pyenv 2>&1 >> $this.pyinstalllog
        } catch {
            $this.createpyapenvir()
        }
    }
    <# -----------------------------------------
     getpythonvers
     get the version number from python
     ------------------------------------------
     Usage: $this.getpythonvers()
    ----------------------------------------- #>   
    [string]getpythonvers(){
        conda activate $this.pyenv 2>&1 >> $this.pyinstalllog
        $l = python -c "from astropath.utilities.version import astropathversion; print(astropathversion)"
        conda deactivate $this.pyenv 2>&1 >> $this.pyinstalllog
        return($l)
    }
}