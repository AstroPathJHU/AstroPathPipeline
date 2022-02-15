<# -------------------------------------------
 testpswarpoctets
 created by: Andrew Jorquera
 Last Edit: 02.10.2022
 --------------------------------------------
 Description
 test if the methods of warpoctets are 
 functioning as intended
 -------------------------------------------#>
#
Class testpswarpoctets {
    #
    [string]$mpath 
    [string]$processloc
    [string]$basepath
    [string]$module = 'warpoctets'
    [string]$slideid = 'M21_1'
    [string]$project = '0'
    [string]$batchid = '8'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$batchbinfile
    [string]$batchreference
    [switch]$fast_test = $true
    #
    testpswarpoctets(){
        #
        $this.launchtests()
        #
    }
    testpswarpoctets($project, $slideid){
        #
        $this.slideid = $slideid
        $this.project = $project
        $this.launchtests
        #
    }
    #
    [void]launchtests(){
        #
        Write-Host '---------------------test ps [warpoctets]---------------------'
        $this.importmodule()
        $task = ($this.project, $this.slideid, $this.processloc, $this.mpath)
        $this.testpswarpoctetsconstruction($task)
        $inp = warpoctets $task
        $this.testprocessroot($inp)
        $this.comparepywarpoctetsinput($inp)
        $this.runpytaskpyerror($inp)
        $this.testlogpyerror($inp)
        $this.runpytaskaperror($inp)
        $this.testlogaperror($inp)
        $this.runpytaskexpected($inp)
        $this.testlogsexpected($inp)
        #$this.CleanupTest($inp)
        Write-Host '.'
    }
    <# --------------------------------------------
    importmodule
    helper function to import the astropath module
    and define global variables
    --------------------------------------------#>
    importmodule(){
        Import-Module $this.apmodule
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.batchbinfile = $PSScriptRoot + '\data\astropath_processing\flatfield\flatfield_melanoma_batches_3_5_6_7_8_9_v2.bin'
        $this.batchreference = $PSScriptRoot + '\data\reference\batchflatfieldcohort\flatfield_TEST.bin'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_warpoctets'))
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
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
        Write-Host '    warpoctets command:'
        Write-Host '   '$pythontask  
        Write-Host '    external log:' $externallog
        Write-Host '    launching task'
        #
        $inp.sample.checkconda()
        etenv $inp.sample.pyenv()
        Invoke-Expression $pythontask *>> $externallog
        exenv
        #
    }
    <# --------------------------------------------
    testpswarpoctetsconstruction
    test that the warpoctets object can be constucted
    --------------------------------------------#>
    [void]testpswarpoctetsconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [warpoctets] constructors started'
        #
        $log = logger $this.mpath $this.module $this.slideid 
        #
        try {
            warpoctets  $task | Out-Null
        } catch {
            Throw ('[warpoctets] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            warpoctets  $task $log | Out-Null
        } catch {
            Throw ('[warpoctets] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [warpoctets] constructors finished'
        #
    }
    <# --------------------------------------------
    testprocessroot
    compare the proccessing root created by the 
    warpoctets object is the same as the one created
    by user defined or known input. Make sure that
    we reference user defined input so that if
    we run the test with an alternative sample
    it will still work.
    --------------------------------------------#>
    [void]testprocessroot($inp){
        #
        Write-Host '.'
        Write-Host 'test processing dir preparation started'
        #
        Write-Host  '   check for an old run and remove it if found'
        $inp.sample.removedir($this.processloc)
        #
        $md_processloc = (
            $this.processloc,
            'astropath_ws',
            $this.module,
            $this.slideid
        ) -join '\'
        #
        if (!([regex]::escape($md_processloc) -contains [regex]::escape($inp.processloc))){
            Write-Host 'meanimage module process location not defined correctly:'
            Write-Host $md_processloc '~='
            Throw ($inp.processloc)
        }
        #
        $inp.sample.CreateNewDirs($inp.processloc)
        #
        if (!(test-path $md_processloc)){
            Throw 'process working directory not created'
        }
        Write-Host 'test processing dir preparation finished'
        #
    }
    <# --------------------------------------------
    comparepywarpoctetsinput
    check that warpoctets input is what is expected
    from the warpoctets module object
    --------------------------------------------#>
    [void]comparepywarpoctetsinput($inp){
        #
        Write-Host '.'
        Write-Host 'compare python [warpoctets] expected input to actual started'
        #
        $md_processloc = ($this.processloc, 'astropath_ws', $this.module, $this.slideid, 'warpoctets') -join '\'
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        [string]$userpythontask = ('warpingcohort',
            $dpath,
            '--shardedim3root', $rpath,
            '--sampleregex', $this.slideid,
            '--flatfield-file',  $this.batchbinfile,
            '--octets-only',
            '--noGPU',
            '--allow-local-edits',
            '--skip-start-finish')
        #
        $inp.getmodulename()
        $pythontask = $inp.getpythontask($dpath, $rpath)
        if (!([regex]::escape($userpythontask) -eq [regex]::escape($pythontask))){
            Write-Host 'user defined and [warpoctets] defined tasks do not match:'  -foregroundColor Red
            Write-Host 'user defined       :' [regex]::escape($userpythontask)'end'  -foregroundColor Red
            Write-Host '[warpoctets] defined:' [regex]::escape($pythontask)'end' -foregroundColor Red
            Throw ('user defined and [warpoctets] defined tasks do not match')
        }
        Write-Host 'python [warpoctets] input matches -- finished'
        #
    }
    <# --------------------------------------------
    runpytaskpyerror
    check that the python task completes correctly 
    when run with the input that will throw a
    python error
    --------------------------------------------#>
    [void]runpytaskpyerror($inp){
        #
        Write-Host '.'
        Write-Host 'test python warpoctets with error input started'
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
        $pythontask = $inp.getpythontask($dpath, $rpath) 
        $pythontask = $pythontask, '--blah' -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) + '.err.log'
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host 'test python warpoctets with error input finished'
        #
    }
    <# --------------------------------------------
    testlogpyerror
    check that the log is parsed correctly
    when run with the input that will throw a
    python error
    --------------------------------------------#>
    [void]testlogpyerror($inp){
        #
        Write-Host '.'
        Write-Host 'test python with error input started'
        #
        $inp.getmodulename()
        $externallog = $inp.ProcessLog($inp.pythonmodulename) + '.err.log'
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($externallog)
        Write-Host '    test log output'
        #
        try {
            $inp.getexternallogs($externallog)
        } catch {
            $err = $_.Exception.Message
            $expectedoutput = 'Error in launching python task'
            if ($err -notcontains $expectedoutput){
                Write-Host $logoutput
                Throw $_.Exception.Message
            }
        }
        #
    }
    <# --------------------------------------------
    runpytaskaperror
    check that the python task completes correctly 
    when run with the input that will throw a
    warpoctets sample error
    --------------------------------------------#>
    [void]runpytaskaperror($inp){
        #
        Write-Host '.'
        Write-Host 'test python warpoctets with error in processing started'
        $inp.sample.CreateDirs($inp.processloc)
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
        $des = $this.processloc, $this.slideid, 'warpoctets' -join '\'
        $pythontask = $inp.pythonmodulename, $dpath, `
        '--shardedim3root',  $rpath, `
        '--sampleregex',  $inp.sample.slideid, `
        '--flatfield-file', $this.batchreference, `
        '--workingdir', $des, `
        '--octets-only --noGPU', $inp.buildpyopts() -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) + '.err.log'
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host 'test python warpoctets with error in processing finished'
        #
    }
    <# --------------------------------------------
    testlogaperror
    check that the log is parsed correctly
    when run with the input that will throw a
    warpoctets sample error
    --------------------------------------------#>
    [void]testlogaperror($inp){
        #
        Write-Host '.'
        Write-Host 'test python with error input started'
        #
        $inp.getmodulename()
        $externallog = $inp.ProcessLog($inp.pythonmodulename) + '.err.log'
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($externallog)
        Write-Host '    test log output'
        #
        try {
            $inp.getexternallogs($externallog)
        } catch {
            $err = $_.Exception.Message
            $expectedoutput = 'detected error in external task'
            if ($err -notcontains $expectedoutput){
                Write-Host $logoutput
                Write-Host $_.Exception.Message
            }
        }
    }
    <# --------------------------------------------
    runpytaskexpected
    check that the python task completes correctly 
    when run with the correct input.
    --------------------------------------------#>
    [void]runpytaskexpected($inp){
        #
        Write-Host '.'
        Write-Host 'test python warpoctets in workflow started'
        $inp.sample.CreateDirs($inp.processloc)
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        Write-Host 'Processvars: '$inp.processvars
        #
        $et_offset_file = $this.basepath,'corrections\best_exposure_time_offsets_Vectra_9_8_2020.csv' -join '\'
        $des = $this.processloc, $this.slideid, 'warpoctets' -join '\'
        $inp.sample.createdirs($des)
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        #$pythontask = $inp.pythonmodulename, $dpath, `
        #'--shardedim3root',  $rpath, `
        #'--sampleregex',  $inp.sample.slideid, `
        #'--flatfield-file',  $this.batchreference, `
        #'--octets-only --noGPU', $inp.buildpyopts() -join ' '
        
        
        #
        $addedargs = #'--workingdir', $des, `
                     '--exposure-time-offset-file', $et_offset_file , `
                     '--initial-pattern-octets','0', `
                     '--principal-point-octets','0', `
                     '--final-pattern-octets','0' -join ' '
        #
        $pythontask = $pythontask, $addedargs -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host 'test python warpoctets in workflow finished'
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
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
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
    <# --------------------------------------------
    testcleanup
    test that the processing directory gets deleted.
    Also remove the 'testing_warpoctets' folder
    for the next run.
    --------------------------------------------#>
    [void]CleanupTest($inp){
        #
        Write-Host '.'
        Write-Host 'test cleanup method started'
        #
        $cleanuppath = (
            $this.processloc,
            'astropath_ws',
            $this.module,
            $this.slideid
        ) -join '\'
        #
        Write-Host '    running cleanup method'
        $inp.cleanup()
        #
        if (Test-Path $cleanuppath) {
            Throw (
                'dir still exists -- cleanup test failed:', 
                $cleanuppath
            ) -join ' '
        }
        Write-Host '    cleanup method complete'
        Write-Host '    delete the testing_warpoctets folder'
        #
        $inp.sample.removedir($this.processloc)
        #
        Write-Host 'test cleanup method finished'
    }
    #
}
#
# launch test and exit if no error found
#
[testpswarpoctets]::new() | Out-Null
exit 0