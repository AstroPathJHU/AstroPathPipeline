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
        ###$this.testvminformconstruction($task)
        $inp = vminform $task
        ###$this.testoutputdir($inp)
        ###$this.testimagelist($inp)
        ###$this.comparevminforminput($inp)
        $this.testkillinformprocess($inp)
        #$this.testruninformexpected($inp)
        #$this.testruninformbatcherror($inp)
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
        $this.setupexpected($inp)
        #
        $inp.StartInForm()
        $inp.KillinFormProcess()
        Write-Host 'inform process successfully ended - starting inform again'
        #
        $inp.StartInForm()
        $inp.WatchBatchInForm()
        $inp.CheckErrors()
        $inp.KillinFormProcess()
        #
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        $inp.sample.CreateNewDirs($this.outpath)
        Write-Host 'test kill inform process finished'
        #
    }
    <# --------------------------------------------
    testruninformexpected
    test that inform is run correctly when run 
    with the correct input.
    --------------------------------------------#>
    [void]testruninformexpected($inp){
        #
        Write-Host '.'
        Write-Host 'test run on inform with expected outcome started'
        #
        $this.setupexpected($inp)
        #
        while(($inp.err -le 5) -AND ($inp.err -ge 0)){
            $inp.StartInForm()
            $inp.WatchBatchInForm()
            $inp.CheckErrors()
            if (($inp.err -le 5) -and ($inp.err -gt 0)){
                $inp.sample.warning("Task will restart. Attempt "+ $inp.err)
            } elseif ($inp.err -gt 5){
                Throw "Could not complete task after 5 attempts"
            } elseif ($inp.err -eq -1){
                $inp.sample.info("inForm Batch Process Finished Successfully")
            }
        }
        $inp.sample.CreateNewDirs($inp.sample.flatwim3folder())
        #
        Write-Host 'test run on inform with expected outcome finished'
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
        Write-Host '    copying reference im3 file to flatw folder:' $this.referenceim3
        $inp.sample.copy($this.referenceim3, $inp.sample.flatwim3folder())
        #
        $inp.CreateOutputDir()
        $inp.DownloadFiles()
        $inp.CreateImageList()
        #
    }
    <# --------------------------------------------
    testruninformbatcherror
    check that the inform task completes correctly 
    when run with the input that will throw a
    inform batch error
    --------------------------------------------#>
    [void]testruninformbatcherror($inp){
        #
        Write-Host '.'
        Write-Host 'test run on inform with batch error started'
        #
        $this.setupbatcherror($inp)
        #
        while(($inp.err -le 5) -AND ($inp.err -ge 0)){
            $inp.StartInForm()
            $inp.WatchBatchInForm()
            $inp.CheckErrors()
            if (($inp.err -le 5) -and ($inp.err -gt 0)){
                $inp.sample.warning("Task will restart. Attempt "+ $inp.err)
            } elseif ($inp.err -gt 5){
                Throw "Could not complete task after 5 attempts"
            } elseif ($inp.err -eq -1){
                $inp.sample.info("inForm Batch Process Finished Successfully")
            }
        }
        $inp.sample.removefile($this.sample.flatwim3folder(), '.im3')
        #
        Write-Host 'test run on inform with batch error finished'
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
        $referenceim3s = $this.basepath, 'M21_1\im3\Scan1\MSI' -join '\'
        Write-Host '    copying reference im3 files to flatw folder:'
        $inp.sample.copy($referenceim3s, $inp.sample.flatwim3folder(), '.im3', 30)
        #
        $inp.CreateOutputDir()
        $inp.DownloadFiles()
        $inp.CreateImageList()
        #
    }
    #
}
#
# launch test and exit if no error found
#
[testvminform]::new() | Out-Null
exit 0