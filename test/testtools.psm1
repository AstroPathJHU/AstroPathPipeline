<# -------------------------------------------
testtools
Benjamin Green
Last Edit: 01.18.2022
--------------------------------------------
Description
test tools
-------------------------------------------#>
#
Class testtools{
    #
    [string]$mpath = "$PSScriptRoot\data\astropath_processing"
    [string]$processloc
    [string]$basepath = "$PSScriptRoot\data"
    [string]$module = 'meanimage'
    [string]$class
    [string]$slideid = 'M21_1'
    [string]$project = '0'
    [string]$batchid = '8'
    [string]$apmodule = "$PSScriptRoot/../astropath"
    [string]$pytype = 'sample'
    [switch]$dryrun = $false 
    [string]$batchreferencefile
    [string]$pybatchflatfieldtest = 'melanoma_batches_3_5_6_7_8_9_v2'
    [string]$slidelist = '"L1_1|M148|M206|M21_1|M55_1|YZ71|ZW2|MA12"'
    [string]$slideid2 = 'M55_1'
    #
    testtools(){
        $this.importmodule()
    }
    #
    testtools($module){
        $this.module = $module
        $this.importmodule()
    }
    #
    testtools($project, $slideid){
        $this.slideid = $slideid
        $this.project = $project
        $this.importmodule()
    }
    #
    testtools($project, $slideid, $dryrun){
        $this.slideid = $slideid
        $this.project = $project
        $this.mpath = '\\bki04\astropath_processing'
        $this.dryrun = $true
        $this.importmodule()
    }
    #
    testtools($project, $slideid, $batchid, $dryrun){
        $this.slideid = $slideid
        $this.project = $project
        $this.batchid = $batchid
        $this.mpath = '\\bki04\astropath_processing'
        $this.dryrun = $true
        $this.importmodule()
    }
    #
    <# --------------------------------------------
    importmodule
    helper function to import the astropath module
    and define global variables
    --------------------------------------------#>
    importmodule(){
        #
        Write-Host '---------------------test ps ['$this.class']---------------------'
        Write-Host "."
        Write-Host 'importing module ....'
        Import-Module $this.apmodule -Global
        $this.processloc = $this.uncpath(("$PSScriptRoot\test_for_jenkins\testing_" + $this.module))
        $batchreferncetestpath = $this.processloc, $this.slideid -join '\'
        $this.batchreferencefile = ($batchreferncetestpath + '\flatfield_TEST.bin')
        $this.basepath = $this.uncpath($this.basepath)
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
        $inp.sample.start(($this.class+'-test'))
        Write-Host "    ["$this.class"] command:"
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
    #
    <# --------------------------------------------
    testprocessroot
    compare the proccessing root created by the 
    warpoctets object is the same as the one created
    by user defined or known input. Make sure that
    we reference user defined input so that if
    we run the test with an alternative sample
    it will still work.
    --------------------------------------------#>
    [void]testprocessroot($inp, $type){
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
    #
    [void]compareinputs($userpythontask, $pythontask){
        #
        if (!([regex]::escape($userpythontask) -eq [regex]::escape($pythontask))){
            Write-Host 'user defined and ['$this.class'] defined tasks do not match:'  -foregroundColor Red
            Write-Host 'user defined       :' [regex]::escape($userpythontask)'end'  -foregroundColor Red
            Write-Host '['$this.class'] defined:' [regex]::escape($pythontask)'end' -foregroundColor Red
            Throw ('user defined and ['+$this.class+'] defined tasks do not match')
        }
        Write-Host 'python ['$this.class'] input matches -- finished'
        #
    }
    #
    [void]addwarpoctetsdep($inp){
        if ($this.dryrun){
            return
        }
        write-Host '    adding warping directories'
        #
        $sor = $this.basepath, 'reference', 'warpingcohort',
        'M21_1-all_overlap_octets.csv' -join '\'
        #
        $inp.getslideidregex()
        Write-Host '    Slides:' $inp.batchslides
        #
        $inp.batchslides | ForEach-Object{
            $des = $this.basepath, $_, 'im3', 'warping', 'octets' -join '\'
            Write-Host '   '$des 
            Write-Host '   '$_
            $inp.sample.copy($sor, $des)
            if ($_ -notmatch 'M21_1'){
                rename-item ($des + '\M21_1-all_overlap_octets.csv') `
                    ($_ + '-all_overlap_octets.csv') -EA stop
            }
        }
    }
    #
    [void]removewarpoctetsdep($inp){
        #
        if ($this.dryrun){
            return
        }
        #
        write-Host '    Removing warping directories'
        #
        $inp.getslideidregex()
        Write-Host '    Slides:' $inp.batchslides
        #
        $inp.batchslides | ForEach-Object{
            $des = $this.basepath, $_, 'im3', 'warping' -join '\'
            $inp.sample.removedir($des)
        }
    }
    #
    [array]getmoduletask($inp){
        #
        $taskname = $this.class
        $inp.getmodulename()
        $dpath = $inp.sample.basepath
        $rpath = '\\' + $inp.sample.project_data.fwpath
        #
        $inp.getslideidregex($this.class)
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        $externallog = $inp.processlog($taskname)
        #
        return @($pythontask, $externallog)
        #
    }
    #
    <# --------------------------------------------
    runpytaskpyerror
    check that the python task completes correctly 
    when run with the input that will throw a
    python error
    --------------------------------------------#>
    [void]runpytaskpyerror($inp){
        #
        Write-Host '.'
        Write-Host 'test python ['$this.class'] with error input started'
        $inp.sample.CreateNewDirs($inp.processloc)
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
        Write-Host 'test python ['$this.class'] with error input finished'
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
        Write-Host 'test python ['$this.class'] LOG with error input started'
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
        Write-Host 'test python ['$this.class'] LOG with error input finished'
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
        Write-Host 'test python ['$this.class'] LOG with error in processing started'
        #
        $inp.getmodulename()
        $externallog = $inp.ProcessLog($inp.pythonmodulename) + '.err.log'
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
        #
        Write-Host 'test python ['$this.class'] LOG with error in processing finished'
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
        Write-Host 'test python ['$this.class'] LOG in workflow started'
        $inp.getmodulename()
        if ($this.class -match 'batch'){
            $inp.getslideidregex()
        }
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        Write-Host '    open log output'
        $logoutput = $inp.sample.GetContent($externallog)
        Write-Host '    test log output'
        #
        try {
            $inp.getexternallogs($externallog)
        } catch {
            if (
                $logoutput -match ' 250 are needed to run all three sets of fit groups!'
            ){
                Write-Host '    test run passed'
            } else{
                Write-Host '   '$logoutput
                Throw $_.Exception.Message
            }
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
        Write-Host 'test python ['$this.class'] LOG in workflow finished'
        #
    }
    #
    <# --------------------------------------------
    testlogsexpected
    check that the log is parsed correctly when
    run with the correct input.
    --------------------------------------------#>
    [void]testlogsexpectedapid($inp){
        #
        Write-Host '.'
        Write-Host 'test python ['$this.class'] LOG in workflow without apid started'
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
        Write-Host 'test python ['$this.class'] LOG in workflow without apid finished'
        #
    }
    #
    <# --------------------------------------------
    testlogsexpected
    check that the log is parsed correctly when
    run with the correct input.
    --------------------------------------------#>
    [void]testlogsexpectednoxml($inp){
        #
        Write-Host '.'
        Write-Host 'test python ['$this.class'] LOG in workflow without apid started'
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
        Write-Host 'test python ['$this.class'] LOG in workflow without apid finished'
        #
    }
}