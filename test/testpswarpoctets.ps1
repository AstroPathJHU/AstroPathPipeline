using module .\testtools.psm1
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
Class testpswarpoctets : testtools {
    #
    [string]$module = 'warpoctets'
    [string]$class = 'warpoctets'
    #
    testpswarpoctets(): base() {
        #
        $this.launchtests()
        #
    }
    testpswarpoctets($project, $slideid): base($project, $slideid) {
        #
        $this.launchtests
        #
    }
    #
    [void]launchtests(){
        #
        $this.testpswarpoctetsconstruction($this.task)
        $inp = warpoctets $this.task
        $this.testprocessroot($inp, $true)
        $this.testcorrectionfile($inp)
        $this.comparepywarpoctetsinput($inp)
        $this.runpytaskpyerror($inp, $true)
        $this.testlogpyerror($inp)
        $this.buildtestflatfield($inp)
        $this.runpytaskaperror($inp)
        $this.testlogaperror($inp)
        $this.setupsample($inp)
        $this.runpytaskexpected($inp)
        $this.testlogsexpected($inp)
        $this.CleanupTest($inp)
        $inp.sample.finish(($this.module+'-test'))
        $this.testgitstatus($inp.sample)
        Write-Host '.'
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
            warpoctets $task | Out-Null
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
    comparepywarpoctetsinput
    check that warpoctets input is what is expected
    from the warpoctets module object
    --------------------------------------------#>
    [void]comparepywarpoctetsinput($inp){
        #
        Write-Host '.'
        Write-Host 'compare python [warpoctets] expected input to actual started'
        #
        $md_processloc = ($this.processloc, 'astropath_ws', $this.module,
            $this.slideid,'warpoctets') -join '\'
        #
        $batchbinfile = $this.mpath + '\flatfield\flatfield_melanoma_batches_3_5_6_7_8_9_v2.bin'
        #
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $taskname = ('warping', $this.pytype) -join ''
        #
        [string]$userpythontask = ($taskname,
            $dpath,
            $this.slideid, #'--sampleregex',
            '--shardedim3root', $rpath,
            '--flatfield-file',  $batchbinfile,
            $inp.gpuopt(),
            '--no-log',
            '--allow-local-edits',
            '--skip-start-finish')
        #
        Write-Host '***Running get module name'
        $inp.getmodulename()
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath)
        #
        $this.compareinputs($userpythontask, $pythontask)
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
        $inp.sample.CreateNewDirs($inp.processloc)
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
        $des = $this.processloc, $this.slideid, 'warpoctets' -join '\'
        $addedargs = " --workingdir $des"
        #
        $pythontask = $inp.('getpythontask' + $this.pytype)($dpath, $rpath) 
        $pythontask = ($pythontask -replace `
            [regex]::escape($inp.sample.pybatchflatfieldfullpath()), 
            $this.batchreferencefile) + $addedargs
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) + '.err.log'
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host 'test python warpoctets with error in processing finished'
        #
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
        $inp.sample.CreateNewDirs($inp.processloc)
        $rpath = $this.processloc, $this.slideid, 'rpath' -join '\'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
        #ff_img = read_image_from_layer_files(ff_file,*(dims),dtype=np.float64)
        #write_image_to_file(ff_img,output_dir/ff_file.name)
        $des = $this.processloc, $this.slideid, 'warpoctets' -join '\'
        $addedargs = (
            ' --workingdir', $des
        ) -join ' '
        #
        $pythontask = $inp.('getpythontask' + $this.pytype)($dpath, $rpath) 
        $pythontask = ($pythontask -replace `
            [regex]::escape($inp.sample.pybatchflatfieldfullpath()), 
            $this.batchreferencefile) + $addedargs
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename)
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        $p2 = (
            $this.basepath,'\',
            $this.slideid,
            '\im3\meanimage\',
            $this.slideid,
            '-background_thresholds.csv'
        ) -join ''
        $inp.sample.removefile($p2)
        #
        Write-Host 'test python warpoctets in workflow finished'
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
        #
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
try {
    [testpswarpoctets]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0