using module .\testtools.psm1
<# -------------------------------------------
 testvmcomponentinform
 created by: Andrew Jorquera
 Last Edit: 12.13.2022
 --------------------------------------------
 Description
 test if the methods of vmcomponentinform are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsvmcomponentinform : testtools {
    #
    [string]$module = 'vmcomponentinform'
    [string]$outpath = "C:\Users\Public\BatchProcessing"
    [string]$referenceim3
    [string]$protocolcopy
    [string]$placeholder
    [switch]$jenkins = $false
    [switch]$versioncheck = $false
    [string]$class = 'vmcomponentinform'
    [string]$informantibody = 'Component'
    [string]$informproject = 'Component_08.ifr'
    #
    testpsvmcomponentinform() : base(){
        #
        $this.launchtests()
        #
    }
    testpsvmcomponentinform($jenkins) : base(){
        #
        $this.jenkins = $true
        $this.launchtests()
        #
    }
    #
    [void]launchtests(){
        #
        $this.testvmcomponentinformconstruction($this.task)
        $inp = vmcomponentinform $this.task
        $this.setupjenkinspaths($inp)
        $this.testoutputdir($inp)
        $this.comparevmcomponentinforminput($inp)
        $this.runinformexpected($inp)
        $this.testlogexpected($inp)
        $this.cleanprotocol($inp)
        Write-Host '.'
        #
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
    testvmcomponentinformconstruction
    test that the vmcomponentinform object can be constucted
    --------------------------------------------#>
    [void]testvmcomponentinformconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [vmcomponentinform] constructors started'
        try {
            vmcomponentinform $task | Out-Null
        } catch {
            Throw ('[vmcomponentinform] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        Write-Host 'test [vmcomponentinform] constructors finished'
        #
    }
    <# --------------------------------------------
    setupjenkinspaths
    set up output paths for when tests are being 
    run on jenkins
    --------------------------------------------#>
    [void]setupjenkinspaths($inp){
        
        if ($this.jenkins) {
            $this.outpath = $this.basepath + '\..\test_for_jenkins\BatchProcessing'
            $inp.outpath = $this.basepath + '\..\test_for_jenkins\BatchProcessing'
            $inp.informoutpath = $this.outpath + '\' + $this.informantibody + '_0'
            $inp.image_list_file = $this.outpath + '\image_list.tmp'
            $inp.informprocesserrorlog =  $this.outpath + "\informprocesserror.log"
            $inp.processvars[0] = $this.outpath
            $inp.processvars[1] = $this.outpath
            $inp.processvars[2] = $this.outpath
        }
        $this.protocolcopy = $this.basepath + '\..\test_for_jenkins\testing_vmcomponentinform'
        $inp.islocal = $false
        $inp.inputimagepath = $inp.outpath + '\' + $inp.sample.slideid + '\im3\flatw'
        $this.placeholder = $this.basepath + '\..\test_for_jenkins\testing_vmcomponentinform'
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
        Write-Host '    saving initial protocol'
        Write-Host ('    copying from ' + $inp.algpath + ' to ' + $this.protocolcopy)
        $inp.sample.copy($inp.algpath, $this.protocolcopy)
        #
        Write-Host '    saving flatwim3 placeholder'
        $placeholderfile = $inp.sample.flatwim3folder() + '\placeholder.txt'
        Write-Host ('    copying from ' + $placeholderfile + ' to ' + $this.placeholder)
        $inp.sample.copy($placeholderfile, $this.placeholder)
        #
        $md_processloc = (
            $this.outpath,
            ($this.informantibody + '_0')
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
    comparevmcomponentinforminput
    check that vmcomponentinform input is what is expected
    from the vmcomponentinform module object
    --------------------------------------------#>
    [void]comparevmcomponentinforminput($inp){
        #
        Write-Host '.'
        Write-Host 'compare [vmcomponentinform] expected input to actual started'
        #
        $informoutpath = $this.outpath, ($this.informantibody + '_0') -join '\'
        $md_imageloc = $this.outpath, 'image_list.tmp' -join '\'
        $algpath = $this.basepath, 'tmp_inform_data', 'Project_Development', 'Component', $this.informproject -join '\'
        $informpath = '"'+"C:\Program Files\Akoya\inForm\" + $this.informvers + "\inForm.exe"+'"'
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
        $this.compareinputs($userinformtask, $informtask)
        #
    }
    <# --------------------------------------------
    runinformexpected
    test that inform is run correctly when run 
    with the correct input.
    --------------------------------------------#>
    [void]runinformexpected($inp){
        #
        if ($this.jenkins) {
            return
        }
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
        $this.referenceim3 = $inp.sample.im3folder() + '\M21_1_[45093,13253].im3'
        Write-Host '    copying reference im3 file to flatw folder:' $this.referenceim3
        $inp.sample.copy($this.referenceim3, $inp.sample.flatwim3folder())
        #
        $inp.CreateOutputDir()
        $inp.DownloadFiles()
        $inp.inputimageids = $null
        $inp.CreateImageList()
        $inp.CheckExportOptions()
        #
    }
    <# --------------------------------------------
    testlogexpected
    check that the log is parsed correctly when
    run with the correct input. 
    --------------------------------------------#>
    [void]testlogexpected($inp) {
        #
        if ($this.jenkins) {
            return
        }
        #
        Write-Host '.'
        Write-Host 'test inform logs with expected outcome started'
        #
        if (!(Test-Path $inp.informoutpath)) {
            Throw 'no output, error in inform task'
        }
        $files = Get-ChildItem $inp.informoutpath
        if (!$files.name -match 'component_data.tif') {
            Throw 'component data failed to create'
        }
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
    cleanprotocol
    clean protocol for future tests
    --------------------------------------------#>
    [void]cleanprotocol($inp) {
        #
        Write-Host '.'
        Write-Host 'starting clean protocol'
        'returning initial protocol'
        $savedalg = $this.protocolcopy + '\' + $inp.alg
        Write-Host ('    copying from ' + $savedalg + ' to ' + $inp.algpath + '\..')
        $inp.sample.copy($savedalg, ($inp.algpath + '\..'))
        $inp.sample.removefile($savedalg)
        Write-Host 'finished return initial protocol'
        #
        Write-Host 'returning initial flatw placeholder'
        $savedplaceholder = $this.placeholder + '\placeholder.txt'
        Write-Host ('    copying from ' + $savedplaceholder + ' to ' + $inp.sample.flatwim3folder())
        $inp.sample.copy($savedplaceholder, ($inp.sample.flatwim3folder()))
        $inp.sample.removefile($savedplaceholder)
        Write-Host 'finished return initial flatwplaceholder'
        #
    }
    #
}
#
# launch test and exit if no error found
#
#[testpsvmcomponentinform]::new() | Out-Null

#
# add $jenkins parameter to constructor if testing on jenkins
#
[testpsvmcomponentinform]::new($jenkins) | Out-Null

exit 0