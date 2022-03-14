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
    [string]$referenceim3
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
        #$this.testvminformconstruction($task)
        $inp = vminform $task
        <#
        $this.testoutputdir($inp)
        $this.testimagelist($inp)
        $this.comparevminforminput($inp)
        $this.testkillinformprocess($inp)
        #>
        $this.runinformexpected($inp)
        $this.testlogexpected($inp)
        $this.runinformbatcherror($inp)
        $this.testlogbatcherror($inp)
        Write-Host '.'
    }
    <# --------------------------------------------
    importmodule
    helper function to import the astropath module
    and define global variables
    --------------------------------------------#>
    importmodule(){
        Import-Module $this.apmodule -global
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_vminform'))
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
        $this.referenceim3 = $this.basepath, 'M21_1\im3\Scan1\MSI\M21_1_[45093,13253].im3' -join '\'
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
    comparepathsexclude
    helper function that uses the copy utils
    file hasher to quickly compare two directories
    excludes certain files types to avoid failed
    comparisons due to timestamps in files
    --------------------------------------------#>
    [void]comparepathsexclude($patha, $pathb, $inp, $filetype){
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
        $lista = Get-ChildItem $patha -recurse -exclude $filetype -file
        $listb = Get-ChildItem $pathb -recurse -exclude $filetype -file
        $hasha = $inp.sample.FileHasher($lista)
        $hashb = $inp.sample.FileHasher($listb)
        $comparison = Compare-Object -ReferenceObject $($hasha.Values) `
                -DifferenceObject $($hashb.Values)
        if ($comparison){
            Write-Host 'Comparison:' $comparison
            Throw 'file contents do not match'
        }
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
        #$log = logger $this.mpath $this.module $this.slideid 
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
        #
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
        $md_imageloc = ($this.outpath, 'image_list.tmp') -join '\'
        Write-Host '    creating image list file:' $md_imageloc
        #
        $inp.DownloadFiles()
        $inp.CreateImageList()
        if (!([regex]::escape($md_imageloc) -contains [regex]::escape($inp.image_list_file))){
            Write-Host 'vminform module process location not defined correctly:'
            Write-Host $md_imageloc '~='
            Throw ($inp.image_list_file)
        }
        #
        if (!(test-path $md_imageloc)){
            Throw 'process working directory not created'
        }
        #
        $inp.sample.CreateNewDirs($this.outpath)
        Write-Host 'test create image list finished'
        #
    }
    <# --------------------------------------------
    comparevminforminput
    check that vminform input is what is expected
    from the vminform module object
    --------------------------------------------#>
    [void]comparevminforminput($inp){
        #
        Write-Host '.'
        Write-Host 'compare [vminform] expected input to actual started'
        #
        $informoutpath = $this.outpath, $this.procedure -join '\'
        $md_imageloc = $this.outpath, 'image_list.tmp' -join '\'
        $algpath = $this.basepath, 'tmp_inform_data', 'Project_Development', $this.algorithm -join '\'
        $informpath = '"'+"C:\Program Files\Akoya\inForm\" + $this.informver + "\inForm.exe"+'"'
        $informprocesserrorlog =  $this.outpath, "informprocesserror.log" -join '\'
        #
        $processoutputlog =  $this.outpath + '\processoutput.log'
        $arginput = ' -a',  $algpath, `
                    '-o',  $informoutpath, `
                    '-i', $md_imageloc -join ' '
        #
        [string]$userinformtask = $informpath,
                                  '-NoNewWindow',
                                  '-RedirectStandardError', $informprocesserrorlog,
                                  '-PassThru',
                                  '-ArgumentList',  $arginput,
                                  '*>>', $processoutputlog -join ' '
        #
        $informtask = $inp.getinformtask()
        #
        if (!([regex]::escape($userinformtask) -eq [regex]::escape($informtask))){
            Write-Host 'user defined and [vminform] defined tasks do not match:'  -foregroundColor Red
            Write-Host 'user defined      :' [regex]::escape($userinformtask)'end'  -foregroundColor Red
            Write-Host '[vminform] defined:' [regex]::escape($informtask)'end' -foregroundColor Red
            Throw ('user defined and [vminform] defined tasks do not match')
        }
        Write-Host '[vminform] input matches -- finished'
        #
    }
    <# --------------------------------------------
    testkillinformprocess
    test that the inform path can be found and
    that it can be shut down correctly
    --------------------------------------------#>
    [void]testkillinformprocess($inp){
        #
        Write-Host '.'
        Write-Host 'test kill inform process started'
        #
        $this.setupbatcherror($inp)
        $inp.StartInForm()
        Write-Host '    inform process started - running kill inform'
        $inp.KillinFormProcess()
        Write-Host '    inform process ended - emptying output directory'
        $log = $inp.sample.GetContent($inp.informprocesserrorlog)
        Write-Host '    inform process log output:'
        Write-Host $log
        #
        Write-Host '    starting inform again'
        $this.setupbatcherror($inp)
        $inp.StartInForm()
        Write-Host '    inform process started - waiting'
        $inp.WatchBatchInForm()
        $log = $inp.sample.GetContent($inp.informprocesserrorlog)
        Write-Host '    inform process log output:'
        Write-Host $log
        #
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        $inp.sample.CreateNewDirs($this.outpath)
        #
        Write-Host 'test kill inform process finished'
        #
    }
    <# --------------------------------------------
    runinformexpected
    test that inform is run correctly when run 
    with the correct input.
    --------------------------------------------#>
    [void]runinformexpected($inp){
        #
        Write-Host '.'
        Write-Host 'run on inform with expected outcome started'
        #
        $this.setupexpected($inp)
        #
        $inp.StartInForm()
        $inp.WatchBatchInForm()
        #
        Write-Host 'run on inform with expected outcome finished'
        #
    }
    <# --------------------------------------------
    setupexpected
    helper function to help setup the processing
    directory to be able to start inform session
    with expected outcome
    --------------------------------------------#>
    [void]setupexpected($inp) {
        #
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        $inp.sample.CreateNewDirs($this.outpath)
        #
        Write-Host '    copying reference im3 file to flatw folder:' $this.referenceim3
        $inp.sample.copy($this.referenceim3, $inp.sample.flatwim3folder())
        #
        $inp.KillinFormProcess()
        $inp.CreateOutputDir()
        $inp.DownloadFiles()
        $inp.CreateImageList()
        #
    }
    <# --------------------------------------------
    testlogexpected
    check that the log is parsed correctly when
    run with the correct input. 
    --------------------------------------------#>
    [void]testlogexpected($inp) {
        #
        Write-Host '.'
        Write-Host 'test inform logs with expected outcome started'
        #
        Write-Host '    comparing output with reference'
        $reference = $inp.sample.basepath + '\reference\vminform\expected'
        $excluded = @('*.log', '*.ifp')
        $this.comparepathsexclude($reference, $inp.informoutpath, $inp, $excluded)
        Write-Host '    compare successful'
        #
        Write-Host '    Inform Batch Log:' $inp.informbatchlog
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($inp.informbatchlog)
        Write-Host '    test log output'

        $completestring = 'Batch process is completed'
        if ($logoutput -match $completestring) {
            $errormessage = $logoutput.Where({$_ -match $completestring}, 'SkipUntil')
            Write-Host '    Error message:'
            $inp.sample.error(($errormessage | Select-Object -skip 1))
        }
        else {
            throw 'error in inform task - batch process did not complete'
        }
        $inp.sample.CreateNewDirs($this.outpath)
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        Write-Host 'test inform logs with expected outcome finished'
        #
    }
    <# --------------------------------------------
    runinformbatcherror
    check that the inform task completes correctly 
    when run with the input that will throw a
    inform batch error
    --------------------------------------------#>
    [void]runinformbatcherror($inp){
        #
        Write-Host '.'
        Write-Host 'run on inform with batch error started'
        #
        $this.setupbatcherror($inp)
        #
        $inp.StartInForm()
        $inp.WatchBatchInForm()
        #
        Write-Host 'run on inform with batch error finished'
        #
    }
    <# --------------------------------------------
    setupbatcherror
    helper function to help setup the processing
    directory to be able to start inform session
    with batch error outcome
    --------------------------------------------#>
    [void]setupbatcherror($inp) {
        #
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        $inp.sample.CreateNewDirs($this.outpath)
        #
        $referenceim3s = $this.basepath, 'M21_1\im3\Scan1\MSI' -join '\'
        Write-Host '    copying reference im3 files to flatw folder'
        $inp.sample.copy($referenceim3s, $inp.sample.flatwim3folder(), '.im3', 30)
        #
        $inp.KillinFormProcess()
        $inp.CreateOutputDir()
        $inp.DownloadFiles()
        $inp.CreateImageList()
        #
    }
    <# --------------------------------------------
    testlogbatcherror
    check that the log is parsed correctly
    when run with the input that will throw an
    inform batch error.
    writes error log to the main sample log,
    skipping the non-error first line
    --------------------------------------------#>
    [void]testlogbatcherror($inp) {
        #
        Write-Host '.'
        Write-Host 'test inform logs with batch error started'
        #
        Write-Host 'comparing output with reference'
        $reference = $inp.sample.basepath + '\reference\vminform\batcherror'
        $excluded = @('*.log', '*.ifp')
        $this.comparepathsexclude($reference, $inp.informoutpath, $inp, $excluded)
        Write-Host '    compare successful'
        #
        Write-Host '    Inform Batch Log:' $inp.informbatchlog
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($inp.informbatchlog)
        Write-Host '    test log output'

        $completestring = 'Batch process is completed'
        if ($logoutput -match $completestring) {
            $errormessage = $logoutput.Where({$_ -match $completestring}, 'SkipUntil')
            Write-Host '    Error message:'
            $inp.sample.error(($errormessage | Select-Object -skip 1))
        }
        else {
            throw 'error in inform task - batch process did not complete'
        }
        $inp.sample.CreateNewDirs($this.outpath)
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        Write-Host 'test inform logs with batch errors finished'
        #
    }
    #
}
#
# launch test and exit if no error found
#
[testvminform]::new() | Out-Null
exit 0