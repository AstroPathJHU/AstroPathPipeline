<# -------------------------------------------
 testpswarpfits
 Benjamin Green, Andrew Jorquera
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the methods of warpfits are 
 functioning as intended
 -------------------------------------------#>
 Class testpswarpfits{
    #
    [string]$mpath 
    [string]$processloc
    [string]$basepath
    [string]$module = 'meanimage'
    [string]$batchid = '6'
    [string]$project = '1'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$slideid
    [switch]$dryrun = $false
    #
    testpswarpfits(){
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_warpfits'))
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
        $this.launchtests()
    }
    #
    testpswarpfits($dryrun){
        #
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
        Write-Host '---------------------test ps [warpfits]---------------------'
        $this.importmodule()
        $task = ($this.project, $this.batchid, $this.processloc, $this.mpath)
        #$this.testpswarpfitsconstruction($task)
        $inp = batchwarpfits $task  
        #$this.testprocessroot($inp)
        #$this.testslidelist($inp)
        #$this.testshreddatim($inp)
        #$this.runpywarpfitsexpectedall($inp)
        $this.runpywarpfitsexpectedbatch($inp)
        $this.testlogsexpected($inp)
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
    runpytesttask
    helper function to run the python task provided
    and export it to the exteral log provided
    --------------------------------------------#>
    [void]runpytesttask($inp, $pythontask, $externallog){
        #
        $inp.sample.start($this.module)
        Write-Host '    command:'
        Write-Host '   '$pythontask  
        Write-Host '    external log:' $externallog
        Write-Host '    launching task'
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
    [void]testpswarpfitsconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [warpfits] constructors started'
        #
        $log = logger $this.mpath $this.module $this.batchid $this.project 
        #
        try {
            batchwarpfits  $task | Out-Null
        } catch {
            Throw ('[warpfits] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            meanimage  $task $log | Out-Null
        } catch {
            Throw ('[meanimage] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [warpfits] constructors finished'
        #
    }
    <#---------------------------------------------
    testprocessroot
    ---------------------------------------------#>
    [void]testprocessroot($inp){
        #
        Write-Host "."
        Write-Host 'test processing root started'
        Write-Host '    raw path:' $inp.processloc
        Write-Host '    data path:' $inp.processvars[0]
        Write-Host '    data.dat export path:' $inp.processvars[1]
        if (!(test-path $inp.processloc)){
            Throw ('did not make processloc: ' + $inp.processloc)
        }
        Write-Host 'test processing root finished'
        #
    }
    <#---------------------------------------------
    testshreddatim
    ---------------------------------------------#>
    [void] testshreddatim($inp){
        #
        Write-Host "."
        Write-Host 'test shred dat on images started'
        Write-Host '    get slide list from one slideid:' $this.slideid
        Write-Host '    open image keys text'
        $image_keys_file = $this.getkeysloc()
        $image_keys = $inp.sample.GetContent($image_keys_file)
        #
        Write-Host '    get keys for this file'
        $images= $inp.getslidekeypaths('M10_2', $image_keys)
        Write-Host '    keys:'
        Write-Host '   '$images
        #
        if ($this.dryrun){
            $inp.shreddat('M10_2', $images)
            Write-Host '    get keys for all slides in the batch'
            $inp.getwarpdats()
        }
        #
        write-host 'get warp dats finished'
        #
    }
    <#---------------------------------------------
    runpywarpfitsexpected
    ---------------------------------------------#>
    [void]runpywarpfitsexpectedall($inp){
        #
        Write-Host "."
        Write-Host 'test py task started for all slides'
        #
        $taskname = 'batchwarpfits'
        $inp.getmodulename()
        $dpath = $inp.processvars[0]
        $rpath = $inp.processvars[1]
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
        }  else {
            Write-Host '   '$pythontask
            Write-Host '   '$externallog
        }
        #
        Write-Host 'test py task finished'
        #
    }
    <#---------------------------------------------
    runpywarpfitsexpected
    ---------------------------------------------#>
    [void]runpywarpfitsexpectedbatch($inp){
        #
        Write-Host "."
        Write-Host 'test py task started for a batch'
        #
        $taskname = 'batchwarpfits'
        $inp.getmodulename()
        $dpath = $inp.processvars[0]
        $rpath = $inp.processvars[1]
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
        }  else {
            Write-Host '   '$pythontask
            Write-Host '   '$externallog
        }
        #
        Write-Host 'test py task finished'
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
        Write-Host 'test python expected log output started'
        $inp.getmodulename()
        $inp.getslideidregex()
        $externallog = $inp.ProcessLog('batchwarpkeys') 
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($externallog)
        Write-Host '    test log output'
        #
        try {
            $inp.getexternallogs($externallog)
        } catch {
            Write-Host '   '$logoutput
            Throw $_.Exception.Message
        }
        #
        Write-Host 'test python expected log output finished'
        #
    }
 }
 #
 [testpswarpfits]::new() | Out-NUll
 exit 0


