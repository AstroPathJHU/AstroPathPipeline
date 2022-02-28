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
    [string]$module = 'batchwarpkeys'
    [string]$batchid = '6'
    [string]$project = '1'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$slideid
    [switch]$dryrun = $false
    #
    testpsbatchwarpkeys(){
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_warpkeys'))
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
        $this.launchtests()
    }
    #
    testpsbatchwarpkeys($dryrun){
        $this.mpath = '\\bki04\astropath_processing'
        $this.processloc = '\\bki08\e$'
        $this.basepath = '\\bki04\Clinical_Specimen'
        $this.batchid = '6'
        $this.project = '1'
        $this.slideid = 'M10_2'
        $this.dryrun = $true
        $this.launchtests()
        #
    }
    #
    launchtests(){
        Write-Host '---------------------test ps [batchwarpkeys]---------------------'
        $this.importmodule()
        $task = ($this.project, $this.batchid, $this.processloc, $this.mpath)
        $this.testpsbatchwarpkeysconstruction($task)
        $inp = batchwarpkeys $task  
        $this.testprocessroot($inp)
        #$this.runpywarpkeysexpectedall($inp)
        $this.runpywarpkeysexpectedbatch($inp)
        $this.testlogsexpected($inp)
        $inp.sample.finish(($this.module+'-test'))
        Write-Host '.'
    }
    <# --------------------------------------------
    importmodule
    helper function to import the astropath module
    and define global variables
    --------------------------------------------#>
    importmodule(){
        Write-Host "."
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
    runpytesttask
    helper function to run the python task provided
    and export it to the exteral log provided
    --------------------------------------------#>
    [void]runpytesttask($inp, $pythontask, $externallog){
        #
        $inp.sample.start(($this.module+'-test'))
        Write-Host '    warp keys command:'
        Write-Host '   '$pythontask  
        Write-Host '    external log:' $externallog
        Write-Host '    launching task'
        #
        $pythontask = $pythontask -replace '\\','/'
        #
        if ($inp.sample.isWindows()){
            $inp.sample.checkconda()
            etenv $inp.sample.pyenv()
            Invoke-Expression $pythontask *>> $externallog
            exenv
        } else{
            Invoke-Expression $pythontask *>> $externallog
        }
        #
    }
    <# --------------------------------------------
    testpswarpfitsconstruction
    test that the meanimage object can be constucted
    --------------------------------------------#>
    [void]testpsbatchwarpkeysconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [batchwarpkeys] constructors started'
        #
        $log = logger $this.mpath $this.module -batchid:$this.batchid -project:$this.project 
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
    #
    [void]testprocessroot($inp){
        Write-Host '.'
        Write-host 'test creating sample dir started'
        Write-Host '    process location:' $inp.processloc
        $testprocloc = $this.basepath + '\warping\Batch_' + ($this.batchid).PadLeft(2,'0')
        if (!($inp.processloc -contains $testprocloc)){
            Write-Host '    test process loc:' $testprocloc
            Throw 'test process loc does not match [batchwarpkeys] loc'
        }
        Write-Host '    slide id:' $inp.sample.slideid
        Write-host 'test creating sample dir finished'
    }
    #
    [void]runpywarpkeysexpectedall($inp){
        #
        Write-Host "."
        Write-Host 'test py task started for all slides'
        #
        $inp.getmodulename()
        $dpath = $inp.sample.basepath
        $rpath = '\\' + $inp.sample.project_data.fwpath
        $taskname = 'batchwarpkeys'
        $inp.all = $true # uses all slide from the cohort, 
        #   output goes to the mpath\warping\octets folder
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
        if ($this.dryrun){
            $this.runpytesttask($inp, $pythontask, $externallog)
        } else {
            Write-Host '   '$pythontask
            Write-Host '   '$externallog
        }
        #
        Write-Host 'test py task finished'
        #
    }
    #
    [void]runpywarpkeysexpectedbatch($inp){
        #
        Write-Host "."
        Write-Host 'test py task started for [batchwarpkeys] expected output started'
        #
        $taskname = 'batchwarpkeys'
        $inp.getmodulename()
        $dpath = $inp.sample.basepath
        $rpath = '\\' + $inp.sample.project_data.fwpath
        $inp.getslideidregex()
        $inp.sample.createNewdirs(($inp.processloc+ '\octets'))
        #
        Write-Host $inp.sample.pybatchflatfieldfullpath()
        if (!(Test-Path $inp.sample.pybatchflatfieldfullpath())){
            Throw ('flatfield file does not exist: ' + 
                $inp.sample.pybatchflatfieldfullpath())
        }
        #
        $inp.getbatchwarpoctets()
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        $externallog = $inp.processlog($taskname)
        #
        if ($this.dryrun){
            $this.runpytesttask($inp, $pythontask, $externallog)
        } else {
            Write-Host '   '$pythontask
            Write-Host '   '$externallog
        }
        #
        Write-Host 'test py task started for [batchwarpkeys] expected output finished'
        #
    }
    <# --------------------------------------------
    testlogsexpected
    check that the log is parsed correctly when
    run with the correct input.
    --------------------------------------------#>
    [void]testlogsexpected($inp){
        #
        Write-Host '.'
        Write-Host 'test py task started for [batchwarpkeys] LOG expected output started'
        $inp.getmodulename()
        $inp.getslideidregex()
        $externallog = $inp.ProcessLog('batchwarpkeys') 
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($externallog)
        Write-Host '    test log output'
        Write-Host '   '$logoutput
        #
        try {
            $inp.getexternallogs($externallog)
        } catch {
            Write-Host '   '$logoutput
            Throw $_.Exception.Message
        }
        #
        Write-Host 'test py task started for [batchwarpkeys] LOG expected output finished'
        #
    }
#
}
#
[testpsbatchwarpkeys]::new($dryrun) | Out-Null
exit 0

