<# -------------------------------------------
 testvminform
 created by: Andrew Jorquera
 Last Edit: 02.17.2022
 --------------------------------------------
 Description
 test if the methods of vminform are 
 functioning as intended
 -------------------------------------------#>
#
Class testvminform {
    #
    [string]$mpath 
    [string]$processloc
    [string]$basepath
    [string]$module = 'vminform'
    [string]$slideid = 'M21_1'
    [string]$procedure = 'CD8'
    [string]$algorithm = 'CD8_Phenotype.ifp'
    [string]$informver = '2.4.8'
    [string]$outpath = "C:\Users\Public\BatchProcessing"
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    #
    testvminform(){
        #
        $this.launchtests()
        #
    }
    testvminform($project, $slideid){
        #
        $this.slideid = $slideid
        $this.project = $project
        $this.launchtests
        #
    }
    #
    [void]launchtests(){
        #
        Write-Host '---------------------test ps [vminform]---------------------'
        $this.importmodule()
        $task = ($this.basepath, $this.slideid, $this.procedure, $this.algorithm, $this.informver, $this.mpath)
        ###$this.testvminformconstruction($task)
        $inp = vminform $task
        ###$this.testoutputdir($inp)
        ###$this.testimagelist($inp)
        $this.rundispatcher($inp)


        #$this.comparepywarpoctetsinput($inp)
        #$this.runpytaskaperror($inp)
        #$this.testlogaperror($inp)
        #$this.runpytaskexpected($inp)
        #$this.testlogsexpected($inp)
        #$this.testcleanup($inp)
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
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_vminform'))
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
    testvminformconstruction
    test that the vminform object can be constucted
    --------------------------------------------#>
    [void]testvminformconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [vminform] constructors started'
        #
        $log = logger $this.mpath $this.module $this.slideid 
        #
        try {
            vminform $task | Out-Null
        } catch {
            Throw ('[vminform] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            warpoctets  $task $log | Out-Null
        } catch {
            Throw ('[warpoctets] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [vminform] constructors finished'
        #
    }
    <# --------------------------------------------
    testoutputdir
    compare the output directory root created by the 
    vminform object is the same as the one created
    by user defined or known input. Make sure that
    we reference user defined input so that if
    we run the test with an alternative sample
    it will still work.
    --------------------------------------------#>
    [void]testoutputdir($inp){
        #
        Write-Host '.'
        Write-Host 'test create output directory started'
        #
        $md_processloc = (
            $this.outpath,
            $this.procedure
        ) -join '\'
        #
        $inp.CreateOutputDir()
        if (!([regex]::escape($md_processloc) -contains [regex]::escape($inp.informoutpath))){
            Write-Host 'vminform module process location not defined correctly:'
            Write-Host $md_processloc '~='
            Throw ($inp.informoutpath)
        }
        #
        if (!(test-path $md_processloc)){
            Throw 'process working directory not created'
        }
        Write-Host 'test create output directory finished'
        #
    }
    <# --------------------------------------------
    testimagelist
    compare the image list created by the 
    vminform object is the same as the one created
    by user defined or known input. Make sure that
    we reference user defined input so that if
    we run the test with an alternative sample
    it will still work.
    --------------------------------------------#>
    [void]testimagelist($inp){
        #
        Write-Host '.'
        Write-Host 'test create image list started'
        #
        $md_imageloc = (
            $this.outpath,
            'image_list.tmp'
        ) -join '\'
        #
        $inp.sample.copy()
        $inp.DownloadIm3()
        $inp.CreateImageList()
        Write-Host 'flatwim3folder:' $inp.sample.flatwim3folder()
        if (!([regex]::escape($md_imageloc) -contains [regex]::escape($inp.image_list_file))){
            Write-Host 'vminform module process location not defined correctly:'
            Write-Host $md_imageloc '~='
            Throw ($inp.image_list_file)
        }
        #
        if (!(test-path $md_imageloc)){
            Throw 'process working directory not created'
        }
        Write-Host 'test create image list finished'
        #
    }
    #
    [void]rundispatcher($inp){
        $cred = Get-Credential -Message "Provide a user name (domain\username) and password"
        $dis = Dispatcher($this.mpath, $this.module, $this.project, $this.slideid, 'NA', $cred)

    }
    #
    [void]TestIntializeWorkerList($dis){
        Write-Host 'Starting worker list tests'
        #
        $dis.InitializeWorkerlist()
        #
        if(!(($dis.workers.module | Sort-Object | Get-Unique) -contains 'vminform')){
            Throw 'Work List not appropriately defined'
        }
        #
        $dis.GetRunningJobs()
        if ($dis.workers.count -ne 2){
            Throw 'Some workers tagged as running when they are not'
        }
        #
        $this.StartTestJob()
        $dis.GetRunningJobs()
        #
        $currentworker = $dis.workers[0]
        $jobname = $dis.defjobname($currentworker)
        #
        $j = get-job -Name $jobname
        if (!($j)){
            Throw 'orphaned task monitor failed to launch'
        }
        #
        start-sleep -s (1*65)
        #
        if(!((get-job -Name $jobname).State -match 'Completed')){
             Throw 'orphaned task monitor did not close correctly'
        }
        #
        Write-Host 'Passed worker list tests'
        #
    }
    #
    [void]StartTestJob($dis){
        $currentworker = $dis.workers[0]
        $creds = $dis.GetCreds()  
        $currentworkerip = $dis.defcurrentworkerip($currentworker)
        $jobname = $dis.defjobname($currentworker)

         $myscriptblock = {
                param($username, $password, $currentworkerip, $workertasklog)
                psexec -i -nobanner -accepteula -u $username -p $password \\$currentworkerip `
                    powershell -noprofile -noexit -executionpolicy bypass -command "Start-Sleep -s (1*60)" `
                    *>> $workertasklog
            }
        #
        $myparameters = @{
            ScriptBlock = $myscriptblock
            ArgumentList = $creds[0], $creds[1], $currentworkerip, $dis.workertasklog($jobname)
            name = ($jobname + '-test')
            }
        #
        Start-Job @myparameters
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
            $this.slideid,
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
    [void]testcleanup($inp){
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
[testvminform]::new() | Out-Null
exit 0