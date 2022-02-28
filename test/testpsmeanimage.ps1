<# -------------------------------------------
 testpsmeanimage
 Benjamin Green, Andrew Jorquera
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the methods of meanimage are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsmeanimage {
    #
    [string]$mpath 
    [string]$processloc
    [string]$basepath
    [string]$module = 'meanimage'
    [string]$slideid = 'M21_1'
    [string]$project = '0'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$pytype = 'sample'
    #
    testpsmeanimage(){
        #
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
        $this.launchtests()
        #
    }
    #
    testpsmeanimage($project, $slideid){
        #
        $this.slideid = $slideid
        $this.project = $project
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
        $this.launchtests()
        #
    }
    #
    testpsmeanimage($project, $slideid, $dryrun){
        #
        $this.slideid = $slideid
        $this.project = $project
        $this.mpath = '\\bki04\astropath_processing'
        $this.launchtests()
        #
    }
    #
    [void]launchtests(){
        #
        Write-Host '---------------------test ps [meanimage]---------------------'
        $this.importmodule()
        $task = ($this.project, $this.slideid, $this.processloc, $this.mpath)
        $this.testpsmeanimageconstruction($task)
        $inp = meanimage $task 
        $this.testprocessroot($inp)
        $this.testcleanupbase($inp)
        $this.comparepymeanimageinput($inp)
        $this.runpytaskpyerror($inp)
        $this.testlogpyerror($inp)
        $this.runpytaskaperror($inp)
        $this.testlogaperror($inp)
        $this.runpytaskexpected($inp)
        $this.testlogsexpected($inp)
        # $this.runpytaskexpectedapid($inp)
        # $this.testlogsexpectedapid($inp)
        $this.runpytaskexpectednoxml($inp)
        $this.testlogsexpectednoxml($inp)
        $this.testreturndatapy($inp)
        $this.testmasks($inp)
        $this.testcleanup($inp)
        $inp.sample.finish(($this.module+'test'))
        Write-Host '.'
    }
    <# --------------------------------------------
    importmodule
    helper function to import the astropath module
    and define global variables
    --------------------------------------------#>
    importmodule(){
        #
        Write-Host "."
        Write-Host 'importing module ....'
        Import-Module $this.apmodule
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_meanimage'))
        #
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
        $inp.sample.start(($this.module+'test'))
        Write-Host '    meanimage command:'
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
    comparepaths
    helper function that uses the copy utils
    file hasher to quickly compare two directories
    --------------------------------------------#>
    [void]comparepaths($patha, $pathb, $inp){
        #
        Write-Host '    Comparing paths:'
        Write-Host '   '$patha
        Write-Host '   '$pathb
        if (!(test-path $patha)){
            Throw ('path does not exist:', $patha -join ' ')
        }
        #
        if (!(test-path $pathb)){
            Throw ('path does not exist:', $pathb -join ' ')
        }
        #
        $lista = Get-ChildItem $patha -file
        $listb = Get-ChildItem $pathb -file
        #
        $hasha = $inp.sample.FileHasher($lista)
        $hashb = $inp.sample.FileHasher($listb)
        $comparison = Compare-Object -ReferenceObject $($hasha.Values) `
                -DifferenceObject $($hashb.Values)
        if ($comparison){
            Throw 'file contents do not match'
        }
        #
    }
    <# --------------------------------------------
    testpsmeanimageconstruction
    test that the meanimage object can be constucted
    --------------------------------------------#>
    [void]testpsmeanimageconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [meanimage] constructors started'
        #
        $log = logger $this.mpath $this.module $this.slideid 
        #
        try {
            meanimage  $task | Out-Null
        } catch {
            Throw ('[meanimage] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            meanimage  $task $log | Out-Null
        } catch {
            Throw ('[meanimage] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [meanimage] constructors finished'
        #
    }
    <# --------------------------------------------
    testprocessroot
    compare the proccessing root created by the 
    meanimage object is the same as the one created
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
    testcleanupbase
    Create a copy of the old results and then 
    run the cleanupbase method. Finally copy the
    results back to return the directory to it's
    orginal state.
    --------------------------------------------#>
    [void]testcleanupbase($inp){
        #
        Write-Host '.'
        Write-Host 'test cleanup base method started'
        #
        Write-Host '   copying old results to a safe location'
        $sor = $this.basepath, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        $des = $this.processloc, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        #
        Write-Host '   source:' $sor
        Write-Host '   destination:' $des
        $inp.sample.copy($sor, $des, '*')
        #
        Write-Host '   running cleanup protocol'
        $inp.cleanupbase()
        #
        if (test-path $sor){
            Throw 'meanimage directory still exists after cleanup'
        }
        #
        Write-Host '   results appear to be cleared'
        #
        Write-Host '   replacing masks'
        #
        $inp.sample.copy($des, $sor, '*')
        $this.comparepaths($des, $sor, $inp)
        #
        $inp.sample.removedir($des)
        #
        Write-Host 'test cleanup base method finished'

    }
    <# --------------------------------------------
    comparepymeanimageinput
    check that meanimage input is what is expected
    from the meanimage module object
    --------------------------------------------#>
    [void]comparepymeanimageinput($inp){
        #
        Write-Host '.'
        Write-Host 'compare python [meanimage] expected input to actual started'
        #
        $md_processloc = (
            $this.processloc,
            'astropath_ws',
            $this.module,
            $this.slideid,
            'meanimage'
        ) -join '\'
        #
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        [string]$userpythontask = (('meanimage', $this.pytype -join ''),
            $dpath, 
            $this.slideid, #'--sampleregex',
            '--shardedim3root', $rpath,
            ' --workingdir', $md_processloc,
            "--njobs '8'",
            '--allow-local-edits',
            '--skip-start-finish')
        #
        $inp.getmodulename()
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath)
        #
        if (!([regex]::escape($userpythontask) -eq [regex]::escape($pythontask))){
            Write-Host 'user defined and [meanimage] defined tasks do not match:'  -foregroundColor Red
            Write-Host 'user defined       :' [regex]::escape($userpythontask)'end'  -foregroundColor Red
            Write-Host '[meanimage] defined:' [regex]::escape($pythontask)'end' -foregroundColor Red
            Throw ('user defined and [meanimage] defined tasks do not match')
        }
        Write-Host 'python [meanimage] input matches -- finished'
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
        Write-Host 'test python meanimage with error input started'
        $inp.sample.CreateNewDirs($inp.processloc)
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath) 
        $pythontask = $pythontask, '--blah' -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) + '.err.log'
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host 'test python meanimage  with error input finished'
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
    meanimagesample error
    --------------------------------------------#>
    [void]runpytaskaperror($inp){
        #
        Write-Host '.'
        Write-Host 'test python meanimage with error in processing started'
        $inp.sample.CreateDirs($inp.processloc)
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath) 
        $pythontask = $pythontask
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) + '.err.log'
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host 'test python meanimage  with error in processing finished'
        #
    }
    <# --------------------------------------------
    testlogaperror
    check that the log is parsed correctly
    when run with the input that will throw a
    meanimagesample error
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
        Write-Host 'test python meanimage in workflow started'
        $inp.sample.CreateDirs($inp.processloc)
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
        $et_offset_file = $this.basepath,'corrections\best_exposure_time_offsets_Vectra_9_8_2020.csv' -join '\'
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath) 
        $des = $this.processloc, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        $inp.sample.createdirs($des)
        #
        $addedargs = '--selectrectangles',
                     '17 18 19 20 23 24 25 26 29 30 31 32 35 36 37 38 39 40', 
                     '--maskroot', $this.processloc,
                     '--exposure-time-offset-file', $et_offset_file -join ' '
                     
        $pythontask = $pythontask, $addedargs -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host 'test python meanimage in workflow finished'
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
        # check that blank lines didn't write to the log
        #
        $loglines = import-csv $inp.sample.mainlog `
            -Delimiter ';' `
            -header 'Project','Cohort','slideid','Message','Date' 
        if ($inp.sample.module -match 'batch'){
            $ID= $inp.sample.BatchID
        } else {
            $ID = $inp.sample.slideid
        }
        #
        $savelog = $loglines |
                    where-object {($_.Slideid -match $ID) -and 
                        ($_.Message -eq '')} |
                    Select-Object -Last 1 
        if ($savelog){
            Throw 'blank log output exists'
        }
        #
        Write-Host 'test python expected log output finished'
        #
    }
    <# --------------------------------------------
    runpytaskexpected
    check that the python task completes correctly 
    when run with the correct input.
    --------------------------------------------#>
    [void]runpytaskexpectedapid($inp){
        #
        Write-Host '.'
        Write-Host 'test python meanimage in workflow without apid started'
        <#
        Write-Host '    removing sampledef file'
        $samplefile = '\sampledef.csv'
        $samplesor = $this.basepath 
        $sampledes = $this.processloc, $this.slideid, 'test' -join '\'
        $sor = ($samplesor + $samplefile)
        $inp.sample.copy($sor, $sampledes)
        $inp.sample.removefile($sor)
        #
        $samplefile1 = '\AstropathSampledef.csv'
        $samplesor1 = $this.basepath, '\astropath_processing' -join '\'
        $sampledes1 = $this.processloc, $this.slideid, 'test\test' -join '\'
        $sor = ($samplesor1 + $samplefile1)
        $inp.sample.copy($sor, $sampledes1)
        $inp.sample.removefile($sor)
        #>
        $inp.sample.CreateNewDirs($inp.processloc)
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
        $et_offset_file = $this.basepath,'corrections\best_exposure_time_offsets_Vectra_9_8_2020.csv' -join '\'
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath) 
        $des = $this.processloc, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        $inp.sample.createdirs($des)
        #
        $addedargs = '--selectrectangles',
                     '17 18 19 20 23 24 25 26 29 30 31 32 35 36 37 38 39 40', 
                     '--maskroot', $this.processloc,
                     '--exposure-time-offset-file', $et_offset_file -join ' '
                     
        $pythontask = $pythontask, $addedargs -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        $this.runpytesttask($inp, $pythontask, $externallog)
        <#
        Write-Host '    putting sampledef file back'
        $sor = ($sampledes + $samplefile)
        $inp.sample.copy($sor, $samplesor)
        $sor = ($sampledes1 + $samplefile1)
        $inp.sample.copy($sor, $samplesor1)
        #>
        Write-Host 'test python meanimage in workflow without apid finished'
        #
    }
    <# --------------------------------------------
    testlogsexpected
    check that the log is parsed correctly when
    run with the correct input.
    --------------------------------------------#>
    [void]testlogsexpectedapid($inp){
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
    runpytaskexpectednoxml
    check that the python task completes correctly 
    when run with the correct input.
    --------------------------------------------#>
    [void]runpytaskexpectednoxml($inp){
        #
        Write-Host '.'
        Write-Host 'test python meanimage in workflow without xml annos started'
        $inp.sample.CreateNewDirs($inp.processloc)
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        #
        Write-Host '    removing annotations from xml file'
        $sorxml = $inp.sample.annotationxml()
        $desxml = $this.processloc, $this.slideid, 'test' -join '\'
        $inp.sample.copy($sorxml, $desxml)
        $xmlfile = $inp.sample.getcontent($sorxml)
        $xmlfile2 = $xmlfile -replace 'Acquired', 'Blah'
        $inp.sample.setfile($sorxml, $xmlfile2)
        #
        $et_offset_file = $this.basepath,'corrections\best_exposure_time_offsets_Vectra_9_8_2020.csv' -join '\'
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath) 
        $des = $this.processloc, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        $inp.sample.createdirs($des)
        #
        $addedargs = '--selectrectangles',
                     '17 18 19 20 23 24 25 26 29 30 31 32 35 36 37 38 39 40', 
                     '--maskroot', $this.processloc,
                     '--exposure-time-offset-file', $et_offset_file -join ' '
                     
        $pythontask = $pythontask, $addedargs -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host '    putting sampledef file back'
        $sorxml = $inp.sample.Scanfolder()
        $inp.sample.copy($desxml, $sorxml, 'annotations.xml')
        #
        Write-Host 'test python meanimage in workflow without apid finished'
        #
    }
    <# --------------------------------------------
    testlogsexpected
    check that the log is parsed correctly when
    run with the correct input.
    --------------------------------------------#>
    [void]testlogsexpectednoxml($inp){
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
    testreturndatapy
    test that the proper output is expored to the 
    meanimage sample directory. The method should
    return the meanimage .bin files and image masking
    files at the working directory to the sample
    level meanimage folder. The original sample 
    directory does not contain the other files. We 
    ignore image masking for now (which was why the
    are added to a different place), then run the 
    method and check that the other files are in the 
    right place. We add this to a try catch with a 
    finally block that deletes the extra files to 
    return the directory to it's original state.
    --------------------------------------------#>
    [void]testreturndatapy($inp){
        Write-Host '.'
        Write-Host 'test py return data started'
        #
        $module_processloc = (
            $this.processloc,
            'astropath_ws',
            $this.module,
            $this.slideid,
            'meanimage'
        ) -join '\'
        #
        $returnpath = (
            $this.basepath,
            $this.slideid, 
            'im3\meanimage'
        ) -join '\'
        #
        Write-Host '    Processing Path: ' $module_processloc
        Write-Host '    Return Path:     ' $returnpath
        #
        try {
            Write-Host '    running return data method'
            $inp.ReturnDataPy()
            Write-Host '    comparing paths'
            $this.comparepaths($module_processloc, $returnpath, $inp)
        } catch {
            Throw $_.Exception.Message
        } finally {
            Write-Host '    removing files from meanimage dir'
            $files = get-childitem $returnpath -file
            foreach ($file in $files){
                $inp.sample.removefile($file)
            }
        }
        #
        Write-Host 'test py return data finished'
    }
    <# --------------------------------------------
    testmasks
    compare the orginal and new masking results.
    --------------------------------------------#>
    [void]testmasks($inp){
        #
        Write-Host '.'
        Write-Host 'test mask output started'
        #
        $masking_processloc = (
            $this.processloc,
            $this.slideid,
            'im3',
            'meanimage',
            'image_masking\*'
        ) -join '\'
        #
        Write-Host '   '$masking_processloc
        #
        $files = get-childitem $masking_processloc -include '*.bin'
        if (!($files)){
            Throw 'masks not created; check process loc'
        }
        #
        Write-Host 'test mask output finished'
        #
    }
    <# --------------------------------------------
    testcleanup
    test that the processing directory gets deleted.
    Also remove the 'testing_meanimage' folder
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
        Write-Host '    delete the testing_meanimage folder'
        #
        $inp.sample.removedir($this.processloc)
        #
        Write-Host 'test cleanup method finished'
    }
}
#
# launch test and exit if no error found
#
[testpsmeanimage]::new() | Out-Null
exit 0