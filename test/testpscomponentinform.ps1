using module .\testtools.psm1
<# -------------------------------------------
 testvmcomponentinform
 created by: Andrew Jorquera
 Last Edit: 12.7.2022
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
    testpsvmcomponentinform($ver, $proj) : base(){
        #
        $this.versioncheck = $true
        $this.informvers = $ver
        $this.task.informvers = $ver
        $this.informproject = $proj
        $this.task.algorithm = $proj
        $this.launchtests()
        #
    }
    #
    [void]launchtests(){
        #
        $this.testvminformconstruction($this.task)
        $inp = vmcomponentinform $this.task
        $this.setupjenkinspaths($inp)
        $this.testoutputdir($inp)
        $this.testimagelist($inp)
        $this.testcheckexportoptions($inp)
        $this.comparevminforminput($inp)
        $this.testkillinformprocess($inp)
        $this.runinformexpected($inp)
        $this.testlogexpected($inp)
        $this.runinformbatcherror($inp)
        $this.testlogbatcherror($inp)
        $this.testpixelconversion($inp)
        $this.testinformoutputfiles($inp)
        $this.testcheckforknownerrors($inp)
        $this.testfindfixableandmerge($inp)
        $this.testcheckexportoptions($inp)
        $this.runversioncheck($inp)
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
        $this.protocolcopy = $this.basepath + '\..\test_for_jenkins\testing_vminform'
        $inp.islocal = $false
        $inp.inputimagepath = $inp.outpath + '\' + $inp.sample.slideid + '\im3\flatw'
        $this.placeholder = $this.basepath + '\..\test_for_jenkins\testing_vminform'
    }
    <# --------------------------------------------
    testvminformconstruction
    test that the vminform object can be constucted
    --------------------------------------------#>
    [void]testvminformconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [vminform] constructors started'
        try {
            vminform $task | Out-Null
        } catch {
            Throw ('[vminform] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
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
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        Write-Host 'test create image list finished'
        #
    }
    
    <# --------------------------------------------
    testcheckexportoptions
    test that the check export options function
    is working as intended
    --------------------------------------------#>
    [void]testcheckexportoptions($inp) {
        #
        Write-Host '.'
        Write-Host 'test check export options started'
        #
        $inp.GetMergeConfigData()
        #
        Write-Host '    checking default export option'
        $checkpath = $inp.sample.basepath + '\reference\vminform\exportoptions\FoxP3_Phenotyping_NE_v4_EC_Default.ifr'
        $inp.needsbinaryseg = $false
        $inp.needscomponent = $false
        $this.checkprotocol($inp, $checkpath)
        Write-Host '    default export type successful'
        #
        Write-Host '    checking binary seg map export option'
        $checkpath = $inp.sample.basepath + '\reference\vminform\exportoptions\FoxP3_Phenotyping_NE_v4_EC_BinaryMaps.ifr'
        $inp.needsbinaryseg = $true
        $inp.needscomponent = $false
        $this.checkprotocol($inp, $checkpath)
        Write-Host '    binary seg map export type successful'
        #
        Write-Host '    checking component export option'
        $checkpath = $inp.sample.basepath + '\reference\vminform\exportoptions\FoxP3_Phenotyping_NE_v4_EC_Component.ifr'
        $inp.needsbinaryseg = $false
        $inp.needscomponent = $true
        $this.checkprotocol($inp, $checkpath)
        Write-Host '    default compoent type successful'
        #
        Write-Host '    checking binary with component export option'
        $checkpath = $inp.sample.basepath + '\reference\vminform\exportoptions\FoxP3_Phenotyping_NE_v4_EC_BinaryWComponent.ifr'
        $inp.needsbinaryseg = $true
        $inp.needscomponent = $true
        $this.checkprotocol($inp, $checkpath)
        Write-Host '    default binary with component type successful'
        #
        $inp.sample.mergeconfig_data = $null
        $inp.GetMergeConfigData()
        Write-Host 'test check export options finished'
    }
    <# --------------------------------------------
    checkprotocol
    helper function to check that the protocol
    export type line has been changed correctly
    --------------------------------------------#>
    [void]checkprotocol($inp, $checkpath) {
        #
        $inp.CheckExportOptions()
        <#Write-Host 'binary:' $inp.needsbinaryseg 'and component:' $inp.needscomponent
        if ((Get-FileHash $inp.algpath).Hash -ne (Get-FileHash $checkpath).Hash) {
            Write-Host '    alg hash:' (Get-FileHash $inp.algpath).Hash
            Write-Host '    check hash:' (Get-FileHash $checkpath).Hash
            $obj = Compare-Object $inp.algpath $checkpath
            Write-Host 'compare:' $obj

            $obj2 = Compare-Object (Get-Content $inp.algpath) (Get-Content $checkpath)
            Write-Host '    compare2:' $obj2
            throw ($inp.algpath + ' != ' + $checkpath)
        }
        #>
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
        $informoutpath = $this.outpath, ($this.informantibody + '_0') -join '\'
        $md_imageloc = $this.outpath, 'image_list.tmp' -join '\'
        $algpath = $this.basepath, 'tmp_inform_data', 'Project_Development', $this.informproject -join '\'
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
    testkillinformprocess
    test that the inform path can be found and
    that it can be shut down correctly
    --------------------------------------------#>
    [void]testkillinformprocess($inp){
        #
        if ($this.jenkins) {
            return
        }
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
        if ($this.jenkins -or $this.versioncheck) {
            return
        }
        #
        Write-Host '.'
        Write-Host 'test inform logs with expected outcome started'
        #
        Write-Host '    comparing output with reference'
        $reference = $inp.sample.basepath + '\reference\vminform\expected'
        $excluded = @('*.log', '*.ifr')
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
        if ($this.jenkins) {
            return
        }
        #
        Write-Host '.'
        Write-Host 'run on inform with batch error started'
        #
        Start-Sleep 20
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
        Write-Host '    copying reference im3 files to flatw folder'
        $inp.sample.copy($inp.sample.im3folder(), $inp.sample.flatwim3folder(), '.im3', 30)
        #
        $inp.CreateOutputDir()
        $inp.DownloadFiles()
        $inp.inputimageids = $null
        $inp.CreateImageList()
        $inp.CheckExportOptions()
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
        if ($this.jenkins -or $this.versioncheck) {
            return
        }
        #
        Write-Host '.'
        Write-Host 'test inform logs with batch error started'
        #
        Write-Host 'comparing output with reference'
        $reference = $inp.sample.basepath + '\reference\vminform\batcherror'
        $excluded = @('*.log', '*.ifr')
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
    <# --------------------------------------------
    testpixelconversion
    test the manual micron to pixel conversion in 
    the inform project file against the checkbox 
    option in inform
    --------------------------------------------#>
    [void]testpixelconversion($inp) {
        #
        if ($this.jenkins -or $this.versioncheck) {
            return
        }
        #
        Write-Host '.'
        Write-Host 'test pixel conversion started'
        #
        $des = $inp.sample.basepath + '\tmp_inform_data\Project_Development\' + $this.informproject
        #
        Write-Host '    testing pixel conversion against micron file'
        $sor = $inp.sample.basepath + '\tmp_inform_data\Project_Development\FoxP3_Phenotyping_NE_v4_EC_Micron_from_inForm.ifr'
        xcopy $sor $des /q /y /z /j /v | Out-Null
        $this.runinformexpected($inp)
        #
        Write-Host '    comparing output with reference'
        $reference = $inp.sample.basepath + '\reference\vminform\coordinatespacetests\pixel_from_inform'
        $excluded = @('*.log', '*.ifr')
        $this.comparepathsexclude($reference, $inp.informoutpath, $inp, $excluded)
        Write-Host '    compare successful'
        #
        Write-Host '    setting project file back to original'
        $sor = $inp.sample.basepath + '\tmp_inform_data\Project_Development\FoxP3_Phenotyping_NE_v4_EC_Original.ifr'
        xcopy $sor $des /q /y /z /j /v | Out-Null
        #
        Write-Host 'test pixel conversion finished'
    }
    <# --------------------------------------------
    testinformoutputfiles
    test that the checking of inform files output
    from the expected outcome works correctly
    --------------------------------------------#>
    [void]testinformoutputfiles($inp) {
        #
        if ($this.jenkins) {
            return
        }
        #
        Write-Host '.'
        Write-Host 'test check inform output files started'
        #
        $inp.GetMergeConfigData()
        #
        Write-Host '    checking default export option'
        $inp.needsbinaryseg = $false
        $inp.needscomponent = $false
        $this.runinformexpected($inp)
        Write-Host '    default export type successful'
        #
        Write-Host '    checking binary seg map export option'
        $checkpath = $inp.sample.basepath + '\reference\vminform\exportoptions\FoxP3_Phenotyping_NE_v4_EC_BinaryMaps.ifr'
        $inp.needsbinaryseg = $true
        $inp.needscomponent = $false
        $this.checkprotocol($inp, $checkpath)
        Write-Host '    binary seg map export type successful'
        #
        Write-Host '    checking component export option'
        $checkpath = $inp.sample.basepath + '\reference\vminform\exportoptions\FoxP3_Phenotyping_NE_v4_EC_Component.ifr'
        $inp.needsbinaryseg = $false
        $inp.needscomponent = $true
        $this.checkprotocol($inp, $checkpath)
        Write-Host '    default compoent type successful'
        #
        Write-Host '    checking binary with component export option'
        $checkpath = $inp.sample.basepath + '\reference\vminform\exportoptions\FoxP3_Phenotyping_NE_v4_EC_BinaryWComponent.ifr'
        $inp.needsbinaryseg = $true
        $inp.needscomponent = $true
        $this.checkprotocol($inp, $checkpath)
        Write-Host '    default binary with component type successful'
        #
        $inp.sample.mergeconfig_data = $null
        #
        $this.runinformexpected($inp)
        Write-Host '    batch process complete'
        Write-Host 'test check inform output files finished'
    }
    <# --------------------------------------------
    testcheckforknownerrors
    test that batch log errors are correctly
    identified and files are sorted to corrupted
    or skipped files
    --------------------------------------------#>
    [void]testcheckforknownerrors($inp) {
        #
        if ($this.jenkins) {
            return
        }
        #
        Write-Host '.'
        Write-Host 'test check for known batch errors started'
        #
        $this.runinformbatcherror($inp)
        $errorlines = (
            '2022-04-09 02:06:26,645 ERROR - Phenotyping problem processing image "M21_1_[41633,16763]": Please segment cells.',
            '2022-04-09 02:06:26,645 ERROR - C:\Users\Public\BatchProcessing\M21_1\im3\flatw\M21_1_[41633,16763].im3:',
            '     Sequence contains no elements',
            '2022-04-09 02:06:26,645 ERROR - Phenotyping problem processing image "M21_1_[47521,11163]": Please segment cells.',
            '2022-04-09 02:06:26,645 ERROR - C:\Users\Public\BatchProcessing\M21_1\im3\flatw\M21_1_[47521,11163].im3:',
            '     Sequence contains no elements',
            '2022-04-08 21:43:49,192 ERROR - Phenotyping problem processing image "M21_1_[40866,11715]": A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond',
            '2022-06-07 15:24:13,680 ERROR - Adaptive Cell Segmentation problem processing image "M21_1_[19572,40984]"',
            '<!>C:/Program Files/Akoya/inForm/2.4.8/AcapellaResources/AcapellaJengaPlugin/Scripts/DetectMembrane.script(112) [Stencil::AssertFlat]: The stencil contains overlapped objects, cannot raise stencil.@prop:flat flag.'
        ) -join "`n"
        $inp.sample.PopFile($inp.informbatchlog, $errorlines)
        $batchlog = $inp.sample.GetContent($inp.informbatchlog)
        $inp.CheckForKnownErrors($batchlog)
        #
        if ($inp.corruptedfiles.length -ne 1) {
            throw 'check batch error failed - incorrect number of corrupted files'
        }
        if ($inp.skippedfiles.length -ne 44) {
            throw 'check batch error failed - incorrect number of skipped files'
        }
        if ($inp.corruptedfiles -notmatch 'M21_1_\[40866,11715\]') {
            throw 'connection error check failed to add corrupted file'
        }
        if (!($inp.skippedfiles -match 'M21_1_\[41633,16763\]' -and $inp.skippedfiles -match 'M21_1_\[47521,11163\]')) {
            throw 'segment cell error check failed to add skipped files'
        }
        if (!($inp.skippedfiles -match 'M21_1_\[19572,40984\]')) {
            throw 'stencil contains overlapped objects check failed to add skipped files'
        }
        #
        $inp.sample.CreateNewDirs($this.outpath)
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        #
        Write-Host 'test check for known batch errors finished'
    }
    <# --------------------------------------------
    testfindfixableandmerge
    create lists for files to skip and rerun to 
    test check for fixable files and merge output
    directories. run inform twice to ensure error 
    loop is working correctly
    --------------------------------------------#>
    [void]testfindfixableandmerge($inp) {
        #
        if ($this.jenkins) {
            return
        }
        #
        Write-Host '.'
        Write-Host 'test find fixable files and merge loop started'
        #
        $inp.err = 0
        $this.runinformbatcherror($inp)
        $rerunerror = '2022-04-08 21:43:49,192 ERROR - Phenotyping problem processing image "M21_1_[45093,13253]": A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond'
        $inp.sample.PopFile($inp.informbatchlog, $rerunerror)
        $batchlog = $inp.sample.GetContent($inp.informbatchlog)
        $inp.CheckInFormOutputFiles()
        $inp.CheckForKnownErrors($batchlog)
        #
        $inp.CheckForFixableFiles()
        if ($inp.err -ne 1) {
            throw 'error number != 1'
        }
        $localflatw = $inp.outpath + '\' + $inp.sample.slideid + '\im3\flatw'
        if ((Split-Path ($inp.sample.listfiles($localflatw, '*')) -Leaf) -ne 'M21_1_[45093,13253].im3') {
            throw 'error in redownloading flatw file'
        }
        Write-Host '    first inform run test complete'
        #
        $this.reruninform($inp)
        if ($inp.err -ne -1) {
            throw 'error in completing inform successfully'
        }
        $inp.sample.info("inForm Batch Process Finished Successfully")
        $inp.MergeOutputDirectories()
        $inp.informoutpath = $this.outpath + "\" + $this.abx + '_0'
        $inp.sample.CreateNewDirs($this.outpath)
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        #
        Write-Host 'test find fixable files and merge loop finished'
    }
    <# --------------------------------------------
    reruninform
    helperfunction to rerun inform without 
    creating new output directory during setup
    --------------------------------------------#>
    [void]reruninform($inp) {
        $inp.CreateOutputDir()
        $inp.CreateImageList()
        $inp.CheckExportOptions()
        $inp.StartInForm()
        $inp.WatchBatchInForm()
        $inp.CheckErrors()
    }
    <# --------------------------------------------
    runversioncheck
    run version check methods
    --------------------------------------------#>
    [void]runversioncheck($inp) {
        if ($this.versioncheck) {
            $this.runinformversioncheck($inp)
            $this.testlogversioncheck($inp)
        }
    }
    <# --------------------------------------------
    runinformversioncheck
    check that the inform task completes correctly 
    when running with different versions
    --------------------------------------------#>
    [void]runinformversioncheck($inp){
        #
        if ($this.jenkins) {
            return
        }
        #
        Write-Host '.'
        Write-Host 'run on inform version check started'
        #
        Start-Sleep 20
        #
        $this.setupversioncheck($inp)
        #
        $inp.StartInForm()
        $inp.WatchBatchInForm()
        #
        Write-Host 'run on inform version check finished'
        #
    }
    <# --------------------------------------------
    setupversioncheck
    helper function to help setup the processing
    directory to be able to start inform session
    checking different versions
    --------------------------------------------#>
    [void]setupversioncheck($inp) {
        #
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        $inp.sample.CreateNewDirs($this.outpath)
        #
        $this.referenceim3 = $inp.sample.basepath + '\reference\vminform\AP0150028_[46839,5637].im3'
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
    testlogversioncheck
    check that the log of inputted version and 
    algorithm matches the log of the default 
    version and algorithm 
    --------------------------------------------#>
    [void]testlogversioncheck($inp) {
        #
        Write-Host '.'
        Write-Host 'test inform logs checking versions started'
        #
        Write-Host '    comparing output with reference'
        $reference = $inp.sample.basepath + '\reference\vminform\versioncheck'
        $excluded = @('*.log', '*.ifr')
        $this.comparepathsexclude($reference, $inp.informoutpath, $inp, $excluded)
        Write-Host '    compare successful'
        #
        Write-Host '    Inform Batch Log:' $inp.informbatchlog
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($inp.informbatchlog)
        Write-Host '    test log output'
        #
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
        Write-Host 'test inform logs checking versions finished'
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
}
#
# launch test and exit if no error found
#
#[testpsvminform]::new() | Out-Null

#
# add $jenkins parameter to constructor if testing on jenkins
#
[testpsvminform]::new($jenkins) | Out-Null

#
# add version and project parameters to constructor to test different versions of inform
#
#[testpsvminform]::new('2.6.0', 'FoxP3_12.27.2021_Phenotyping_NE_overcall_v2.ifr') | Out-Null
exit 0