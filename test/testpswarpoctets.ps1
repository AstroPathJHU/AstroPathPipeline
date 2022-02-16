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
    [string]$pytype = 'sample'
    [string]$batchreferencefile
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
        #$this.testpswarpoctetsconstruction($task)
        $inp = warpoctets $task
        #$this.testprocessroot($inp)
        #$this.comparepywarpoctetsinput($inp)
        #$this.runpytaskpyerror($inp)
        #$this.testlogpyerror($inp)
        #$this.buildtestflatfield($inp)
        #$this.runpytaskaperror($inp)
        #$this.testlogaperror($inp)
        #$this.setupsample($inp)
        #$this.runpytaskexpected($inp)
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
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_warpoctets'))
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
        $batchreferncetestpath = $this.processloc, $this.slideid -join '\'
        $this.batchreferencefile = ($batchreferncetestpath + '\flatfield_TEST.bin')
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
        $md_processloc = (
            $this.processloc,
            'astropath_ws',
            $this.module,
            $this.slideid,
            'warpoctets'
        ) -join '\'
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
            '--noGPU',
            '--no-log',
            '--allow-local-edits',
            '--skip-start-finish')
        #
        $inp.getmodulename()
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath)
        #
        if (!([regex]::escape($userpythontask) -eq [regex]::escape($pythontask))){
            Write-Host 'user defined and [warpoctets] defined tasks do not match:'  -foregroundColor Red
            Write-Host 'user defined        :' [regex]::escape($userpythontask)'end'  -foregroundColor Red
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
        $pythontask = $inp.('getpythontask' + $this.pytype)($dpath, $rpath) 
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
    python error. 
    --------------------------------------------#>
    [void]testlogpyerror($inp){
        #
        Write-Host '.'
        Write-Host 'test python with error input started'
        #
        $inp.getmodulename()
        $externallog = $inp.ProcessLog($inp.pythonmodulename) + '.err.log'
        if (!$externallog){
            Throw 'No external log'
        }
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($externallog)
        if (!$logoutput){
            Throw 'No log output'
        }
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
        $inp.sample.CreateNewDirs($inp.processloc)
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
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
        if (!$externallog){
            Throw 'No external log'
        }
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($externallog)
        if (!$logoutput){
            Throw 'No log output'
        }
        Write-Host '    test log output'
        #
        try {
            $inp.getexternallogs($externallog)
        } catch {
            $err = $_.Exception.Message
            $expectedoutput = 'detected error in external task'
            if ($err -notcontains $expectedoutput){
                Write-Host $logoutput
                Throw $_.Exception.Message
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
    #
    [void]setupsample($inp){
        #
        Write-Host '    copy background thresholds'
        #
        $p1 = ($this.basepath,
            '\reference\meanimage\',
            $this.slideid,
            '-background_thresholds.csv'
        ) -join ''
        #
        $p2 = ($this.basepath,
            $this.slideid,
            'im3\meanimage'
        ) -join '\'
        #
        $inp.sample.copy($p1, $p2)
        #
        Write-Host '    creating mock raw directory'
        #
        $rpath = (
            $this.basepath,
            'raw',
            $this.slideid
        ) -join '\'
        #
        $rfiles = (get-childitem ($rpath+'\*') '*dat').Name
        #
        Write-Host '    Found:' $rfiles.Length ' raw files'
        Write-Host $rfiles
        #
        $dpath = (
            $this.basepath,
            $this.slideid,
            'im3\Scan1\MSI',
            '*'
        ) -join '\'
        $im3files = (get-childitem $dpath '*im3').Name
        #
        Write-Host '    Found:' $im3files.Length ' im3 files'
        #
        $newrpath = $this.processloc, $this.slideid, 'rpath', $this.slideid -join '\'
        $inp.sample.CreateNewDirs($newrpath)
        Write-Host '    New rpath:' $newrpath
        Write-Host '    Matching files'
        #
        foreach($file in $im3files){
            $rfile = $file -replace 'im3', 'Data.dat'
            $newrfile = $rfiles -match [regex]::escape($rfile)
            #
            if (!$newrfile){
                $newrfile = $rpath + '\' + $rfiles[0]
                $inp.sample.copy($newrfile, $newrpath)
                rename-item ($newrpath + '\' + $rfiles[0]) `
                    ($file -replace 'im3', 'Data.dat')
            }
        }
        #
        Write-Host '    copying regular raw files'
        $inp.sample.copy($rpath, $newrpath, '*')
        #
    #
    }
    #
    [void]buildtestflatfield($inp){
        #
        Write-Host '.'
        Write-Host 'build test flatfield started'
        #
        $batchreferencepath = $this.basepath + '\reference\batchflatfieldcohort\flatfield_TEST.bin'
        $batchreferncetestpath = $this.processloc, $this.slideid -join '\'
        $inp.sample.createdirs($batchreferncetestpath)
        #
        $import = 'import pathlib; import numpy as np; from astropath.utilities.img_file_io import read_image_from_layer_files, write_image_to_file' -join ' '
        $task1 = "ff_file = pathlib.Path ('", ($batchreferencepath -replace '\\', '/'), "')" -join '' 
        Write-Host '    Task1:' $task1
        $task2 = "ff_img = read_image_from_layer_files(ff_file,1004,1344,35,dtype=np.float64)" -join ''
        Write-Host '    Task2:' $task2
        $task3 =  "outputdir = pathlib.Path ('", ($batchreferncetestpath -replace '\\', '/'), "')" -join ''
        Write-Host '    Task3:' $task3
        $task4 = "write_image_to_file(ff_img, outputdir/ff_file.name)" -join ''
        Write-Host '    Task4:' $task4
        $task = $import, $task1, $task2, $task3, $task4 -join '; '
        Write-Host '    Task:' $task
        #
        $inp.sample.checkconda()
        conda run -n $inp.sample.pyenv() python -c $task
        if (!(test-path $this.batchreferencefile )){
            Throw 'Batch flatfield reference file failed to create'
        }
        #
        Write-Host 'build test flatfield finished'
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