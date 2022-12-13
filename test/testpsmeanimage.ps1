using module .\testtools.psm1
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
Class testpsmeanimage : testtools {
    #
    [string]$module = 'meanimage'
    [string]$class = 'meanimage'
    #
    testpsmeanimage() : base(){
        #
        $this.launchtests()
        #
    }
    #
    testpsmeanimage($project, $slideid) : base($project, $slideid) {
        #
        $this.launchtests()
        #
    }
    #
    testpsmeanimage($project, $slideid, $dryrun) : base($project, $slideid, $dryrun){
        #
        $this.launchtests()
        #
    }
    #
    [void]launchtests(){
        #
        $this.testpsmeanimageconstruction($this.task)
        $inp = meanimage $this.task 
        $inp.sample.teststatus = $true  
        $this.checkcreatepyenv($inp.sample)
        $this.testprocessroot($inp, $true)
        $this.comparepymeanimageinput($inp)
        $this.runpytaskpyerror($inp, $true)
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
        $inp.sample.finish(($this.module+'-test'))
        $this.testgitstatus($inp.sample)
        Write-Host '.'
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
        $this.compareinputs($userpythontask, $pythontask)
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
        Write-Host 'test python [meanimage] with error in processing started'
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
        Write-Host 'test python [meanimage]  with error in processing finished'
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
        Write-Host 'test python [meanimage] in workflow started'
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
                     '--maskroot', $this.processloc -join ' '
                     
        $pythontask = $pythontask, $addedargs -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host 'test python [meanimage] in workflow finished'
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
        Write-Host 'test python [meanimage] in workflow without apid started'
        #
        Write-Host '    removing sampledef file'
        $samplefile = '\sampledef.csv'
        $samplesor = $this.basepath 
        $sampledes = $this.processloc, $this.slideid, 'test' -join '\'
        $sor = ($samplesor + $samplefile)
        $inp.sample.copy($sor, $sampledes)
        $inp.sample.removefile($sor)
        #
        $samplefile1 = '\AstroPathSampledef.csv'
        $samplesor1 = $this.basepath, '\astropath_processing' -join '\'
        $sampledes1 = $this.processloc, $this.slideid, 'test\test' -join '\'
        $sor = ($samplesor1 + $samplefile1)
        $inp.sample.copy($sor, $sampledes1)
        $inp.sample.removefile($sor)
        #
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
                     '--maskroot', $this.processloc -join ' '
                     
        $pythontask = $pythontask, $addedargs -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host '    putting sampledef file back'
        $sor = ($sampledes + $samplefile)
        $inp.sample.copy($sor, $samplesor)
        $sor = ($sampledes1 + $samplefile1)
        $inp.sample.copy($sor, $samplesor1)
        #
        Write-Host 'test python [meanimage] in workflow without apid finished'
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
        Write-Host 'test python [meanimage] in workflow without xml annos started'
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
        if ($pythontask -notmatch $this.slideid){
            Write-Host '    python task:' $pythontask
            Throw 'annotation file missing test failied; slideid is in python task'
        }
        $des = $this.processloc, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        $inp.sample.createdirs($des)
        #
        $addedargs = '--selectrectangles',
                     '17 18 19 20 23 24 25 26 29 30 31 32 35 36 37 38 39 40', 
                     '--maskroot', $this.processloc -join ' '
                     
        $pythontask = $pythontask, $addedargs -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host '    putting sampledef file back'
        $sorxml = $inp.sample.Scanfolder()
        $inp.sample.copy($desxml, $sorxml, 'annotations.xml')
        #
        Write-Host 'test python [meanimage] in workflow without xml annos finished'
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
        Write-Host 'test py return data paths started'
        #
        $des = $inp.sample.im3mainfolder() + '\meanimage'
        #
        $returnpath = (
            $this.basepath,
            $this.slideid, 
            'im3\meanimage'
        ) -join '\'
        #
        $this.compareinputs($des, $returnpath)
        $sor = $inp.processvars[0] +'\meanimage'
        #
        Write-Host 'test py return data paths finished'
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
        Write-Host 'test cleanup method finished'
    }
}
#
# launch test and exit if no error found
#
try {
    [testpsmeanimage]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0