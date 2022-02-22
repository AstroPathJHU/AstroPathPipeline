<# -------------------------------------------
 testpsbatchwarpkeys
 Benjamin Green, Andrew Jorquera
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the methods of batchwarpkeys are 
 functioning as intended
 -------------------------------------------#>
 Class testpsbatchwarpkeys{
    #
    [string]$mpath 
    [string]$processloc
    [string]$basepath
    [string]$module = 'meanimage'
    [string]$batchid = '6'
    [string]$project = '1'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$slideid
    #
    testpsbatchwarpkeys(){
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_meanimage'))
        $this.launchtests()
    }
    #
    testpsbatchwarpkeys($dryrun){
        $this.mpath = '\\bki04\astropath_processing'
        $this.processloc = '\\bki08\e$'
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
        $this.batchid = '6'
        $this.project = '1'
        $this.slideid = 'M10_2'
        $this.launchtests()
    }
    #
    launchtests(){
        Write-Host '---------------------test ps [batchwarpkeys]---------------------'
        $this.importmodule()
        $task = ($this.project, $this.batchid, $this.processloc, $this.mpath)
        #$this.testpsbatchwarpkeysconstruction($task)
        $inp = batchwarpkeys $task  
        #$this.testprocessroot($inp)
        #$this.testslidelist($inp)
        #$this.testshreddatim($inp)
        #$this.runpywarpkeysexpectedall($inp)
        $this.runpywarpkeysexpectedbatch($inp)
        Write-Host '.'
    }
    <# --------------------------------------------
    importmodule
    helper function to import the astropath module
    and define global variables
    --------------------------------------------#>
    importmodule(){
        Write-Host 'importing module ....'
        Import-Module $this.apmodule
    }
    <# --------------------------------------------
    uncpath
    helper function to convert local paths defined
    by pscriptroot etc. to full unc paths.
    --------------------------------------------#>
    [string]uncpath($str){
        $r = $str -replace( '/', '\')
        if ($r[0] -ne '\'){
            $root = ('\\' + $env:computername+'\'+$r) -replace ":", "$"
        } else{
            $root = $r -replace ":", "$"
        }
        return $root
    }
    <# --------------------------------------------
    testpswarpfitsconstruction
    test that the meanimage object can be constucted
    --------------------------------------------#>
    [void]testpswarpfitsconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [batchwarpkeys] constructors started'
        #
        $log = logger $this.mpath $this.module $this.batchid $this.project 
        #
        try {
            batchwarpkeys  $task | Out-Null
        } catch {
            Throw ('[batchwarpkeys] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            meanimage  $task $log | Out-Null
        } catch {
            Throw ('[meanimage] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [batchwarpkeys] constructors finished'
        #
    }
    [void]testprocessroot($inp){
        Write-Host '.'
        Write-host 'test creating sample dir started'
        Write-Host '    process location:' $inp.processloc
       # $inp.sample.createdirs($inp.processloc)
        $spdir = $inp.sample.mpath + '\warping\octets'
       # $inp.sample.createdirs($spdir)
        write-host '    testing dir:' $spdir
        #
        Write-Host '    slide id:' $inp.sample.slideid
        Write-host 'test creating sample dir finished'
    }
    #
    [void]runpywarpkeysexpectedall($inp){
        #
        $inp.getmodulename()
        $dpath = $inp.sample.basepath
        $rpath = '\\' + $inp.sample.project_data.fwpath
        $taskname = 'batchwarpkeys'
        $inp.all = $true
        $inp.getslideidregex()
        #
        Write-Host $inp.sample.pybatchflatfieldfullpath()
        if (!(Test-Path $inp.sample.pybatchflatfieldfullpath())){
            Throw ('flatfield file does not exist: ' + $inp.sample.pybatchflatfieldfullpath())
        }
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        $externallog = $inp.processlog($taskname)
        #
        # $batchslides = $inp.sample.batchslides.slideid -join '|'
        # $pythontask = $inp.getpythontask($dpath, $rpath, $batchslides)
        #
        Write-Host 'running: '
        Write-Host $pythontask
        Write-Host $externallog
        #
        $inp.sample.checkconda()
        conda activate $inp.sample.pyenv()
        Invoke-Expression $pythontask *>> $externallog
        conda deactivate 
        #
    }
    #
    [void]runpywarpkeysexpectedbatch($inp){
        #
        $warpdirectory = $inp.sample.mpath + '\warping\octets'
        $inp.sample.copy($warpdirectory, ($inp.processloc+'\octets'), '*')
        $inp.sample.CreateNewDirs($warpdirectory)
        #
        $inp.getmodulename()
        $dpath = $inp.sample.basepath
        $rpath = '\\' + $inp.sample.project_data.fwpath
        $taskname = 'batchwarpkeys'
        $inp.all = $false
        $inp.getslideidregex()
        #
        Write-Host $inp.sample.pybatchflatfieldfullpath()
        if (!(Test-Path $inp.sample.pybatchflatfieldfullpath())){
            Throw ('flatfield file does not exist: ' + $inp.sample.pybatchflatfieldfullpath())
        }
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        $externallog = $inp.processlog($taskname)
        #
        # $batchslides = $inp.sample.batchslides.slideid -join '|'
        # $pythontask = $inp.getpythontask($dpath, $rpath, $batchslides)
        #
        Write-Host 'running: '
        Write-Host $pythontask
        Write-Host $externallog
        #
        $inp.sample.checkconda()
        conda activate $inp.sample.pyenv()
        Invoke-Expression $pythontask *>> $externallog
        conda deactivate 
        #
    }
#
}
#
[testpsbatchwarpkeys]::new($dryrun) | Out-Null

