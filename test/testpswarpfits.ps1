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
    #
    testpswarpfits(){
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_meanimage'))
        $this.launchtests()
    }
    #
    testpswarpfits($dryrun){
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
        Write-Host '---------------------test ps [warpfits]---------------------'
        $this.importmodule()
        $task = ($this.project, $this.batchid, $this.processloc, $this.mpath)
        #$this.testpswarpfitsconstruction($task)
        $inp = batchwarpfits $task  
        #$this.testprocessroot($inp)
        #$this.testslidelist($inp)
        #$this.testshreddatim($inp)
        $this.runpywarpfitsexpected($inp)
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
        Write-Host $inp.processloc
        Write-Host $inp.processvars[0]
        Write-Host $inp.processvars[1]
        if (!(test-path $inp.processloc)){
            Throw ('did not make processloc: ' + $inp.processloc)
        }
        Write-Host 'test processing root finished'
        #
    }
    <#---------------------------------------------
    testslidelist
    ---------------------------------------------#>
    [void]testslidelist($inp){
        #
        Write-Host "."
        Write-Host 'test building slide list started'
        Write-Host '    test one batch slidelist'
        #
        $inp.getslideidregex()
        Write-Host '    slides in batch list:'
        Write-Host '   '$inp.batchslides
        Write-Host '    slide id is:' $inp.sample.slideid
        #
        Write-Host '    test all slides slidelist'
        $inp.all = $true
        $inp.getslideidregex()
        Write-Host '    slides in batch list:'
        Write-Host '   '$inp.batchslides
        Write-Host '    slide id is:' $inp.sample.slideid
        Write-Host 'test building slide list finished'
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
        $image_keys_file = $inp.sample.mpath + '\warping\octets\image_keys_needed.txt'
        $image_keys = $inp.sample.GetContent($image_keys_file)
        #
        Write-Host '    get keys for this file'
        $images= $inp.getslidekeypaths('M10_2', $image_keys)
        Write-Host '    keys:'
        Write-Host '   '$images
        #
        $inp.shreddat('M10_2', $images)
        #
        Write-Host '    get keys for all slides in the batch'
        $inp.getwarpdats()
        write-host 'get warp dats finished'
        #
    }
    <#---------------------------------------------
    runpywarpfitsexpected
    ---------------------------------------------#>
    [void]runpywarpfitsexpected($inp){
        #
        Write-Host "."
        Write-Host 'test py task started'
        $taskname = 'batchwarpfits'
        $inp.getmodulename()
        $dpath = $inp.processvars[0]
        $rpath = $inp.processvars[1]
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
        Write-Host '    running: '
        Write-Host $pythontask
        Write-Host $externallog
        #
        $inp.sample.checkconda()
        conda activate $inp.sample.pyenv()
        Invoke-Expression $pythontask *>> $externallog
        conda deactivate 
        #
        Write-Host 'test py task finished'
        #
    }
 }
# $inp.runbatchwarpkeys()
 [testpswarpfits]::new($true) | Out-NUll
 exit 0


