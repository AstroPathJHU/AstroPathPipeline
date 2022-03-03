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
    [string]$informver = '2.6.0'
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
        $this.testoutputdir($inp)
        $this.testimagelist($inp)
        $this.testruninform($inp)
        #$this.testreturndata($inp)
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
        #
        $inp.DownloadIm3()
        $inp.CreateImageList()
        Write-Host '    flatwim3folder:' $inp.sample.flatwim3folder()
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
    [void]testruninform($inp){
        #
        Write-Host '.'
        Write-Host 'test run on inform started'
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
        #
        Write-Host 'test run on inform finished'
        #
    }
    <# --------------------------------------------
    testreturndata
    test that the processing directory gets deleted.
    Also remove the 'testing_warpoctets' folder
    for the next run.
    --------------------------------------------#>
    [void]testreturndata($inp){
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