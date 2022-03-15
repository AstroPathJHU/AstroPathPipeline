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
    [string]$batchid = '8'
    [string]$project = '0'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$slideid = 'M21_1'
    [switch]$dryrun = $false
    [string]$pybatchflatfieldtest = 'melanoma_batches_3_5_6_7_8_9_v2'
    [string]$slidelist = '"L1_1|M148|M206|M21_1|M55_1|YZ71|ZW2|MA12"'
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
        #$this.testpsbatchwarpkeysconstruction($task)
        $inp = batchwarpkeys $task  
        #$this.testprocessroot($inp)
        $this.testwarpkeysinputbatch($inp, 'batch')
        $this.runpywarpkeysexpected($inp, 'batch')
        $this.testlogsexpected($inp, 'batch')
        $this.testcleanup($inp, 'batch')
        #
        $this.testwarpkeysinputbatch($inp, 'all')
        $this.runpywarpkeysexpected($inp, 'all')
        $this.testlogsexpected($inp, 'all')
        $this.testcleanup($inp, 'all')
        $inp.sample.finish(($this.module+'-test'))
        #
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
    [void]addwarpoctetsdep($inp){
        if ($this.dryrun){
            return
        }
        write-Host '    adding warping directories'
        #
        $sor = $this.basepath, 'reference', 'warpingcohort',
        'M21_1-all_overlap_octets.csv' -join '\'
        #
        $inp.getslideidregex()
        Write-Host '    Slides:' $inp.batchslides
        #
        $inp.batchslides | ForEach-Object{
            $des = $this.basepath, $_, 'im3', 'warping', 'octets' -join '\'
            $inp.sample.copy($sor, $des)
            rename-item ($des + '\M21_1-all_overlap_octets.csv') `
                ($_ + '-all_overlap_octets.csv') -EA stop
        }
    }
    #
    [void]removewarpoctetsdep($inp){
        #
        if ($this.dryrun){
            return
        }
        #
        write-Host '    Removing warping directories'
        #
        $inp.getslideidregex()
        Write-Host '    Slides:' $inp.batchslides
        #
        $inp.batchslides | ForEach-Object{
            $des = $this.basepath, $_, 'im3', 'warping' -join '\'
            $inp.sample.removedir($des)
        }
    }
    #
    [array]getmoduletask($inp){
        #
        $taskname = 'batchwarpkeys'
        $inp.getmodulename()
        $dpath = $inp.sample.basepath
        $rpath = '\\' + $inp.sample.project_data.fwpath
        #
        $inp.getslideidregex()
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        $externallog = $inp.processlog($taskname)
        #
        return @($pythontask, $externallog)
        #
    }
    #
    [void]testwarpkeysinputbatch($inp, $type){
        #
        if ($this.dryrun){
            return
        }
        #
        Write-Host "."
        Write-Host 'test for [batchwarpkeys] expected input started' 
        #
        $flatwpath = '\\' + $inp.sample.project_data.fwpath
        $this.addwarpoctetsdep($inp)
        #
        if ($type -contains 'all'){
            $inp.all = $true
            $slides = $this.slidelist
            $wd = ' --workingdir', ($this.mpath +
                 '\warping\Project_' + $this.project) -join ' '
        } else {
            $slides = '"M21_1"'
            $wd = ' --workingdir', ($this.basepath + '\warping\Batch_'+
                 $this.batchid.PadLeft(2,'0')) -join ' '
        }
        #
        Write-Host '    collecting [warpkeys] defined task'
        $task = $this.getmoduletask($inp)
        #
        Write-Host '    collecting [user] defined task'
        $userpythontask = (('warpingcohort',
            $this.basepath, 
            '--shardedim3root', $flatwpath,
            '--sampleregex', $slides,
            '--flatfield-file', ($this.mpath + '\flatfield\flatfield_' +
                         $this.pybatchflatfieldtest + '.bin'),
            '--octets-only --noGPU --no-log --ignore-dependencies --allow-local-edits',
            '--use-apiddef --project', $this.project.PadLeft(2,'0')
        ) -join ' ') + $wd
        #
        Write-Host '[user] defined    :' [regex]::escape($userpythontask)'end'  -foregroundColor Red
        Write-Host '[warpkeys] defined:' [regex]::escape($task[0])'end' -foregroundColor Red
        #
        if (!([regex]::escape($userpythontask) -eq [regex]::escape($task[0]))){
            Write-Host 'user defined and [warpoctets] defined tasks do not match:'  -foregroundColor Red
            Throw ('user defined and [warpoctets] defined tasks do not match')
        }
        #
        $this.removewarpoctetsdep($inp)
        #
    }
    #
    [void]runpywarpkeysexpected($inp, $type){
        #
        Write-Host "."
        Write-Host 'test for [batchwarpkeys] expected output' $type 'slides started'
        #
        if ($type -contains 'all'){
            $inp.all = $true# uses all slide from the cohort, 
            #   output goes to the mpath\warping\octets folder
        }
        #
        $this.addwarpoctetsdep($inp)
        #
        $task = $this.getmoduletask($inp)
        $inp.getbatchwarpoctets()
        $this.runpytesttask($inp, $task[0], $task[1])
        #
        $this.removewarpoctetsdep($inp)
        #
        Write-Host 'test for [batchwarpkeys] expected output' $type ' slides finished'
        #
    }
    <# --------------------------------------------
    testlogsexpected
    check that the log is parsed correctly when
    run with the correct input.
    --------------------------------------------#>
    [void]testlogsexpected($inp, $type){
        #
        Write-Host '.'
        Write-Host 'test py task started for [batchwarpkeys] LOG expected output started'
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
            if (
                $logoutput -match ' 250 are needed to run all three sets of fit groups!'
            ){
                Write-Host '    test run passed'
            } else{
                Write-Host '   '$logoutput
                Throw $_.Exception.Message
            }
        }
        #
        Write-Host 'test py task started for [batchwarpkeys] LOG expected output finished'
        #
    }
    #
    [void]testcleanup($inp, $type){
        #
        if ($this.dryrun){
            return
        }
        #
        Write-Host '.'
        Write-Host 'test cleaning tasks up started'
        #
        if ($type -contains 'all'){
            $inp.all = $true# uses all slide from the cohort, 
            #   output goes to the mpath\warping\octets folder
            $warpingkeysfolder = $this.mpath + '\warping\Project_' + $this.project       
        } else {
            $warpingkeysfolder =  ($this.basepath + 
            '\warping\Batch_'+ $this.batchid.PadLeft(2,'0'))
        }
        #
        Write-Host '    path expected to be removed:' $warpingkeysfolder
        #
        $inp.cleanup()
        #
        if (test-path $warpingkeysfolder){
            Throw 'cleaup did not delete folder, path still exists'
        }
        Write-Host 'test cleaning tasks up finish'
        #
    }
#
}
#
[testpsbatchwarpkeys]::new() | Out-Null
exit 0

