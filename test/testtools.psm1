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
    [string]$testrpath
    [string]$apfile_temp_constant = 'Template.csv'
    [string]$pybatchwarpingfiletest = 'warping_BatchID_08.csv'
    [string]$batchflatfieldgtest = 'BatchID_08'
    [hashtable]$task 
    [string]$informvers = '2.4.8'
    [string]$informantibody = 'CD8'
    [string]$informproject = 'blah.ifr'
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
    <# --------------------------------------------
    importmodule
    helper function to import the astropath module
    and define global variables
    --------------------------------------------#>
    importmodule(){
        #
        Write-Host ('---------------------test ps ['+$this.class+']---------------------')
        Write-Host "."
        Write-Host 'importing module ....'
        Import-Module $this.apmodule -Global
        $this.processloc = $this.uncpath(("$PSScriptRoot\test_for_jenkins\testing_" + $this.module))
        $batchreferncetestpath = $this.processloc, $this.slideid -join '\'
        $this.batchreferencefile = ($batchreferncetestpath + '\flatfield_TEST.bin')
        $this.basepath = $this.uncpath($this.basepath)
        $this.testrpath = $this.processloc, $this.slideid, 'rpath' -join '\'
        $this.verifyAPIDdef()
        $this.updatepaths()
        $this.task = @{project =$this.project; slideid=$this.slideid;
                    processloc=$this.processloc;mpath= $this.mpath;
                    batchid=$this.batchid;module=$this.module;
                    antibody=$this.informantibody; algorithm=$this.informproject;
                    informvers=$this.informvers
                    }
        #
    }
    #
    [void]checkcreatepyenv($tools){
        if ($tools.isWindows()){
            if(!($tools.CheckpyEnvir())){
                #
                Write-Host '.'
                Write-Host 'ap environment does not exist.'
                Write-Host '    creating in' $tools.pyenv()
                Write-Host '    log will be created at:' $tools.pyinstalllog()
                #
                $tools.CreatepyEnvir()
            }
        }
    }
    #
    [void]verifyAPIDdef(){
        #
        $files = get-childitem $this.mpath ('*' + $this.apfile_temp_constant)
        $tools = sharedtools
        #
        $files | ForEach-Object{
            #
            $file = $_.FullName -replace 'Template.csv','.csv'
            if (!(test-path ($file))){
                try{
                    $sor = $_.FullName
                    $des = $file
                    Write-Host '    creating:' $des
                    Write-Host '    from:' $sor
                    $paths_data = $tools.OpencsvFileConfirm($sor)
                    $paths_data | Export-CSV $des
                } catch {
                    Write-Host "Warning: $file not found and could not be created from template"
                }
            }
            #
        }
    }
    #
    [void]updatepaths(){
        #
        if ($this.dryrun){
            return
        }
        #
        $tools = sharedtools
        #
        $cohort_csv_file =  $tools.cohorts_fullfile($this.mpath) 
        $project_data = $tools.OpencsvFileConfirm($cohort_csv_file)
        $p = $this.uncpath($PSScriptRoot)
        #
        if ([regex]::escape($project_data[0].Dpath) -ne [regex]::escape($p)){
            Write-Host '    paths do not match'
            Write-host '   '$project_data[0].Dpath
            Write-host '   '$p
            Write-Host '    UPDATING THE COHORTS & PATHS TABLES'
            $project_data[0].Dpath = $p 
            $project_data[1].Dpath = $p  + '\data'
            $project_data | Export-CSV $cohort_csv_file 
            #
            $paths_csv_file =  $tools.paths_fullfile($this.mpath) 
            $paths_data = $tools.OpencsvFileConfirm($paths_csv_file)
            $paths_data[0].Dpath = $p 
            $paths_data[1].Dpath = $p + '\data'
            $paths_data[0].FWpath = ($p  + '\flatw') -replace '\\\\', ''
            $paths_data[1].FWpath = ($p  + '\data\flatw') -replace '\\\\', ''
            $paths_data | Export-CSV $paths_csv_file
        }
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
        Write-Host ("    ["+$this.class+"] command:")
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
    [void]comparepaths($patha, $pathb, $tools, $type){
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
        $hasha = $tools.FileHasher($lista)
        $hashb = $tools.FileHasher($listb)
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
        write-host '   user defined:' $md_processloc
        write-host '    module defined:' $inp.processloc
        #
        if (!([regex]::escape($md_processloc) -contains [regex]::escape($inp.processloc))){
            Write-Host 'module process location not defined correctly:'
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
        Write-Host 'user defined:' $userpythontask 'end'
        Write-Host ('['+$this.class+'] defined: ' + $pythontask + ' end')
        #       
        if (!([regex]::escape($userpythontask) -eq [regex]::escape($pythontask))){
            Throw ('user defined and ['+$this.class+'] defined tasks do not match')
        }
        Write-Host ('python ['+$this.class+'] input matches -- finished')
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
        $inp.getmodulename()
        $taskname = $inp.pythonmodulename
        $dpath = $inp.sample.basepath
        $rpath =  $this.testrpath
        #
        $inp.getslideidregex($this.class)
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        $externallog = $inp.processlog($taskname)
        #
        return @($pythontask, $externallog)
        #
    }
    <# --------------------------------------------
    runpytaskpyerror
    check that the python task completes correctly 
    when run with the input that will throw a
    python error
    --------------------------------------------#>
    [void]runpytaskpyerror($inp, $type){
        #
        Write-Host '.'
        Write-Host ('test python ['+$this.class+'] with error input started')
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
        Write-Host ('test python ['+$this.class+'] with error input finished')
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
        Write-Host ('test python ['+$this.class+'] with error input started')
        $inp.sample.CreateNewDirs($inp.processloc)
        $this.addwarpoctetsdep($inp)
        $mtask = $this.getmoduletask($inp)
        $pythontask = $mtask[0]
        $externallog = $mtask[1] + '.err.log'
        #
        $pythontask = $pythontask, '--blah' -join ' '
        #
        $this.runpytesttask($inp, $pythontask, $externallog)
        $this.removewarpoctetsdep($inp)
        #
        Write-Host ('test python ['+$this.class+'] with error input finished')
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
        Write-Host ('test python ['+$this.class+'] LOG with error input started')
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
        Write-Host ('test python ['+$this.class+'] LOG with error input finished')
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
        Write-Host ('test python ['+$this.class+'] LOG with error in processing started')
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
        Write-Host ('test python ['+$this.class+'] LOG with error in processing finished')
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
        Write-Host ('test python ['+$this.class+'] LOG in workflow started')
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
        if (!$logoutput){
            Throw 'No log output'
        }
        #
        try {
            $inp.getexternallogs($externallog)
        } catch {
            if (
                $logoutput -match ' 250 are needed to run all three sets of fit groups!' -or
                $logoutput -match ' 0 fits for the'-or
                $logoutput -match 'FINISH: warping'){
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
        Write-Host ('test python ['+$this.class+'] LOG in workflow finished')
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
        Write-Host ('test python ['+$this.class+'] LOG in workflow without apid started')
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
        Write-Host ('test python ['+$this.class+'] LOG in workflow without apid finished')
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
        Write-Host ('test python ['+$this.class+'] LOG in workflow without apid started')
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
        Write-Host ('test python ['+$this.class+'] LOG in workflow without apid finished')
        #
    }
    #
    [void]testcorrectionfile($inp){
        #
        if ($inp.sample.pybatchflatfield()){
            return
        }
        #
        $p2 = $this.mpath + '\AstroPathCorrectionModels.csv'
        #
        $micomp_data = $inp.sample.ImportCorrectionModels($this.mpath)
        $newobj = [PSCustomObject]@{
            SlideID = $inp.sample.slideid
            Project = $inp.sample.project
            Cohort = $inp.sample.cohort
            BatchID = $inp.sample.batchid
            FlatfieldVersion = $this.pybatchflatfieldtest
            WarpingFile = 'None'
        }
        #
        if ($micomp_data.slideid -notmatch $inp.sample.slideid){
            $micomp_data += $newobj
            $micomp_data | Export-CSV $p2 -NoTypeInformation
        }
        #
        $p3 = $this.mpath + '\flatfield\flatfield_'+
            $this.pybatchflatfieldtest + '.bin'
        if (!(test-path $p3)){
            $inp.sample.SetFile($p3, 'blah de blah')
        }
        #
    }
    #
    [void]testcorrectionfile($tool, $tools){
        #
        $p2 = $this.mpath + '\AstroPathCorrectionModels.csv'
        #
        $micomp_data = $tool.ImportCorrectionModels($this.mpath)
        #
        if (!$tool.slideid){
            return
        }
        #
        $newobj = [PSCustomObject]@{
            SlideID = $tool.slideid
            Project = $tool.project
            Cohort = $tool.cohort
            BatchID = $tool.batchid
            FlatfieldVersion = $this.pybatchflatfieldtest
            WarpingFile = 'None'
        }
        #
        if ($tool.slideid -notmatch ($micomp_data.slideid -join '|')){
            $micomp_data += $newobj
            $micomp_data | Export-CSV $p2 -NoTypeInformation
        }
        #
        $p3 = $this.mpath + '\flatfield\flatfield_'+
            $this.pybatchflatfieldtest + '.bin'
        if (!(test-path $p3)){
            $tool.SetFile($p3, 'blah de blah')
        }
        #
        $tool.ImportCorrectionModels($this.mpath, $false)
        #
    }
    #
    [void]setupsample($inp){
        #
        Write-Host '    copy background thresholds'
        #
        $p1 = ($this.basepath, '\reference\meanimage\',
            $this.slideid, '-background_thresholds.csv') -join ''
        #
        $p2 = ($this.basepath, $this.slideid,
            'im3\meanimage') -join '\'
        #
        $inp.sample.copy($p1, $p2)
        #
        $this.createtestraw($inp)
        #
    #
    }
    #
    [void]createtestraw($inp){
        #
        Write-Host '    creating mock raw directory'
        #
        $rpath = ($this.basepath, 'raw', $this.slideid) -join '\'
        #
        $rfiles = (get-childitem ($rpath+'\*') '*dat').Name
        #
        Write-Host '    Found:' $rfiles.Length ' raw files'
        Write-Host $rfiles
        #
        $dpath = ($this.basepath, $this.slideid,
            'im3\Scan1\MSI', '*') -join '\'
        $im3files = (get-childitem $dpath '*im3').Name
        #
        Write-Host '    Found:' $im3files.Length ' im3 files'
        #
        $newtestrpath = $this.testrpath + '\' + $this.slideid
        $inp.sample.CreateNewDirs($newtestrpath)
        #
        Write-Host '    New rpath:' $newtestrpath
        Write-Host '    Matching files'
        #
        foreach($file in $im3files){
            $rfile = $file -replace 'im3', 'Data.dat'
            $newrfile = $rfiles -match [regex]::escape($rfile)
            #
            if (!$newrfile){
                $newrfile = $rpath + '\' + $rfiles[0]
                $inp.sample.copy($newrfile, $newtestrpath)
                rename-item ($newtestrpath + '\' + $rfiles[0]) `
                    ($file -replace 'im3', 'Data.dat')
            }
        }
        #
        Write-Host '    copying regular raw files'
        $inp.sample.copy($rpath, $newtestrpath, '*')
        #
    }
    #
    [void]addoctetpatterns($inp){
        #
        Write-Host '    adding octet patterns'
        if ($this.all){
            $folder = $inp.sample.warpprojectoctetsfolder()
        } else {
            $folder = $inp.sample.warpbatchoctetsfolder()
        }
        #
        $inp.sample.createnewdirs($folder)
        #
        $reffiles = @( 'initial_pattern_octets_selected.csv',
                    'principal_point_octets_selected.csv',
                    'final_pattern_octets_selected.csv',
                    ($this.slideid + '-all_overlap_octets.csv'))
        #
        $reffiles | foreach-object {
            $reffile = $this.basepath, 'reference',
            'warpingcohort', $_ -join '\'
            $inp.sample.copy($reffile, $folder)
        }
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
        $taska = $import, $task1, $task2, $task3, $task4 -join '; '
        Write-Host '    Task:' $taska
        #
        if ($inp.sample.isWindows()){
            $inp.sample.checkconda()
            conda run -n $inp.sample.pyenv() python -c $taska
        } else{
            python -c $taska
        }
        if (!(test-path $this.batchreferencefile )){
            Throw 'Batch flatfield reference file failed to create'
        }
        #
        Write-Host 'build test flatfield finished'
        #
    }
    #
    [void]addtestfiles($sample, $path, $files){
        #
        foreach ($file in $files) {
            #
            if ($file[0] -match '-'){
                $file = $this.slideid + $file
            }
            #
            $fullpath = $path + '\' + $file
            $sample.setfile($fullpath, 'blah de blah')
            #
        }
        #
    }
    #
    [void]addtestfiles($sample, $path, $file, $source){
        #
        $source = $source -replace '\.', ''
        $file = $file -replace '\.', ''
        $sample.getfiles($source, $false) | ForEach-Object{
            $sample.copy($_.FullName, $path)
            if ($sample.($source + 'constant') `
                -notcontains $sample.($file + 'constant')){
                $newname = $_.Name -replace $sample.($source + 'constant'),
                 $sample.($file + 'constant')
                rename-item ($path + '\' + $_.Name) $newname
            }
        }
        #
    }
    #
    [void]removetestfiles($sample, $path, [array]$files){
        #
        foreach ($file in $files) {
            #
            if ($file[0] -match '-'){
                $file = $this.slideid + $file
            }
            #
            $fullpath = $path + '\' + $file
            write-host '    file to remove:' $fullpath
            $sample.removefile($fullpath)
            Write-Host '    file successfully removed:' (!(test-path $fullpath))
            #
        }
        #
    }
    #
    [void]removetestfiles($sample, $path, $file, $source){
        #
        $source = $source -replace '\.', ''
        $file = $file -replace '\.', ''
        $sample.getfiles($source, $false) | ForEach-Object{
            $newname = $_.Name -replace $($source + 'constant'),
                 $($file + 'constant')
            $sample.removefile($path + '\' + $newname)
        }
        #
    }
    #
    [string]aptempfullname($sampletracker, $filetype){
        #
        $filename = $sampletracker.($filetype + '_fullfile')($this.mpath)
        $tempfilename = $filename `
            -replace $sampletracker.apfile_constant, $this.apfile_temp_constant
        return $tempfilename
        #
    }
    #
    [void]testgitstatus($sample){
        #
        write-host '.'
        write-host ('test git status after [' + $this.class + '] started')
        #
        if (!$sample.checkgitstatustest()){
            $gitstatus = git -C $sample.testpackagepath() status
            write-host $gitstatus
            Throw 'git status not empty changes on branch'
        }
        #
        write-host ('test git status after [' + $this.class + '] finished')
        #
    }
    [void]resetvminform($sample){
        $sample.removefile($this.mpath + '\across_project_queues\vminform-queue.csv')
        $sample.copy(($this.mpath + '\vminform-queue.csv'),
         ($this.mpath + '\across_project_queues'))
        $sample.removefile(
            $sample.vmq.localqueuefile.($this.project))
    }
    #
    [void]showtable($table){
        #
        write-host ($table | Format-Table | Out-String)
        #
    }
    #
    [void]savephenotypedata($sampletracker){
        #
        write-host '    saving inform results'
        if ((test-path ($this.processloc + '\tables')) -and
            !(test-path $sampletracker.mergefolder())){
                return 
            }
        #
        $sampletracker.removedir($this.processloc + '\tables')
        $sampletracker.copy($sampletracker.mergefolder(),
            ($this.processloc + '\tables'))
            $sampletracker.removedir($this.processloc + '\Component_Tiffs')
        $sampletracker.copy($sampletracker.componentfolder(),
            ($this.processloc + '\Component_Tiffs'))
        #
    }
    #
    [void]returnphenotypedata($sampletracker){
        #
        write-host '    returning inform results'
        if (test-path ($this.processloc + '\tables')){
            $sampletracker.removedir($sampletracker.mergefolder())
            $sampletracker.copy(($this.processloc + '\tables'),
                $sampletracker.mergefolder())
        }
        #
        if (test-path ($this.processloc + '\Component_Tiffs')){
            $sampletracker.removedir($sampletracker.componentfolder())
            $sampletracker.copy(($this.processloc + '\Component_Tiffs'),
                $sampletracker.componentfolder())
            $sampletracker.removedir($this.processloc)
        }
        #
    }
    #
    [void]addalgorithms($sampletracker){
        #
        $sampletracker.findantibodies()
        foreach ($abx in $sampletracker.antibodies) {
            #
            $sampletracker.vmq.checkfornewtask($this.project, $this.slideid, $abx)
            $sampletracker.vmq.localqueue.($this.project) |    
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } |
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                }
            $sampletracker.vmq.writelocalqueue($this.project)
            #
            $sampletracker.vmq.coalescevminformqueues($this.project)
            #
            $sampletracker.vmq.maincsv | 
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } | 
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                    $_.ProcessingLocation = ''
                }
                #
        }
        #
    }
    #
    [void]setupbatchwarpkeys($sampledb){
        #
        Write-host '    setting up batcharpkeys dependencies for the environment'
        #   
        $sampledb.modules | ForEach-Object {
            $this.addstartlog($sampledb, $_)
            Start-Sleep 2
            $this.addfinishlog($sampledb, $_)
        }
        #
        $sampledb.preparesample($this.slideid)
        $sampletracker = $sampledb
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.meanimagefolder(), $sampletracker.meanimagereqfiles)
        $this.addtestfiles($sampletracker,
            $sampletracker.meanimagefolder(), '-mask_stack.bin')
        #
        $p2 = $sampletracker.micomp_fullfile($this.mpath)
        #
        $micomp_data = $sampletracker.importmicomp($sampletracker.mpath, $false)
        $newobj = [PSCustomObject]@{
            root_dir_1 = $sampletracker.basepath + '\'
            slide_ID_1 = $sampletracker.slideid
            root_dir_2 = $sampletracker.basepath + '\'
            slide_ID_2 = 'blah'
            layer_n = 1
            delta_over_sigma_std_dev = .95
        }
        $micomp_data += $newobj
        #
        $micomp_data | Export-CSV $p2 -NoTypeInformation
        #
        $this.testcorrectionfile($sampletracker, $true)
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpoctetsfolder(), 
            $sampletracker.warpoctetsreqfiles)
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpbatchoctetsfolder(),
            $sampletracker.batchwarpkeysreqfiles)
        <#
        $sampledb.getmodulelogs($false)
        write-host 'newtasks:'
        write-host $sampledb.newtasks 
        #
        $sampledb.preparesample($this.slideid)
        #>
        $sampledb.getmodulelogs($false)
        $sampledb.refreshsampledb('batchwarpkeys', '0')
        #
    }
    #
    [void]setupvminform($sampledb){
        #
        Write-host '    setting up inform dependencies for the environment'
        #
        $sampledb.preparesample($this.slideid)
        $sampletracker = $sampledb
        $sampletracker.teststatus = $true
        #
        $this.addtestfiles($sampletracker, 
        $sampletracker.warpbatchfolder(),
        $sampletracker.batchwarpfitsreqfiles)
        #
        $this.addalgorithms($sampledb)
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.flatwfolder(),
            $sampletracker.imagecorrectionreqfiles[0], 
            $sampletracker.im3constant
        )
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.flatwfolder(),
            $sampletracker.imagecorrectionreqfiles[1], 
            $sampletracker.im3constant
        )
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.flatwim3folder(),
            $sampletracker.imagecorrectionreqfiles[2], 
            $sampletracker.im3constant
        )
        #
        $sampledb.getmodulelogs($false)
        $sampledb.preparesample($this.slideid)
        #
    }
    #
    [void]removesetupvminform($sampledb){
        #
        Write-host '    removing inform dependencies for the environment'
        #
        $sampledb.preparesample($this.slideid)
        $sampletracker = $sampledb
        $sampletracker.teststatus = $true
        #
        $sampletracker.removefile($this.mpath + '\across_project_queues\vminform-queue.csv')
        $sampletracker.copy(($this.mpath + '\vminform-queue.csv'),
         ($this.mpath + '\across_project_queues'))
        $sampletracker.removefile(
            $sampletracker.vmq.localqueuefile.($this.project))
        $sampletracker.removedir($sampletracker.informfolder())
        $sampletracker.removefile($sampletracker.slidelogbase('shredxml'))
        $sampletracker.removefile($sampletracker.slidelogbase('meanimage'))
        $sampletracker.removefile($sampletracker.slidelogbase('vminform'))
        $sampletracker.removefile($sampletracker.mainlogbase('vminform'))
        #
        $this.removetestfiles($sampletracker,
            $sampletracker.meanimagefolder(), $sampletracker.meanimagereqfiles)
        $this.removetestfiles($sampletracker,
            $sampletracker.meanimagefolder(), '-mask_stack.bin')
        #
        $sampletracker.removefile($sampletracker.mainlogbase('batchmicomp'))
        $sampletracker.removefile($sampletracker.mainlogbase('batchflatfield'))
        $sampletracker.removefile($sampletracker.slidelogbase('warpoctets'))
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpoctetsfolder(), 
            $sampletracker.warpoctetsreqfiles)
        #
        $sampletracker.removefile($sampletracker.mainlogbase('batchwarpfits'))
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpbatchfolder(),
            $sampletracker.batchwarpfitsreqfiles)
        #
        $sampletracker.removefile($sampletracker.mainlogbase('batchwarpkeys'))
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpbatchoctetsfolder(), 
            $sampletracker.batchwarpkeysreqfiles)
        #
        $sampletracker.removedir($sampletracker.basepath +'\warping')
        $sampletracker.removedir($sampletracker.warpoctetsfolder())
        $sampletracker.removedir($sampletracker.flatwim3folder())
        $sampletracker.removedir($sampletracker.flatwfolder())
        $sampletracker.removefile($sampletracker.basepath + '\upkeep_and_progress\imageqa_upkeep.csv')
        #
        #
        $p = $this.aptempfullname($sampletracker, 'corrmodels')
        $p2 = $sampletracker.corrmodels_fullfile($this.mpath)
        #
        $sampletracker.removefile($p2)
        $data = $sampletracker.opencsvfile($p)
        $data | Export-CSV $p2  -NoTypeInformation
        #
        $p3 = $sampletracker.mpath + '\flatfield\flatfield_'+$this.pybatchflatfieldtest+'.bin'
        $sampletracker.removefile($p3)
        #
        #
    }
    #
    [void]addproccessedalgorithms($sampledb){
        #
        $sampledb.findantibodies()
        foreach ($abx in $sampledb.antibodies) {
            #
            $sampledb.vmq.checkfornewtask($this.project, $this.slideid, $abx)
            $sampledb.vmq.localqueue.($this.project) |    
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } |
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                }
            $sampledb.vmq.writelocalqueue($this.project)
            #
            $sampledb.vmq.coalescevminformqueues($this.project)
            #
            $sampledb.vmq.maincsv | 
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } | 
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                    $_.ProcessingLocation = 'Processing: bki##'
                }
                #
        }
        #
        $sampledb.vmq.writemainqueue($this.project)
        #
    }
    #
    [void]addfinishlog($sampledb, $cmodule){
        #
        if ($cmodule -match 'vminform'){
            return
        }
        #   
        if ($cmodule -match 'batch') {
            $mess = $this.project, $this.cohort,
                $this.batchid.PadLeft(2,'0'), ('FINISH:' + $cmodule), (Get-Date) -join ';'
        } else {
            $mess = $this.project, $this.cohort,
                $this.slideid, ('FINISH:' + $cmodule), (Get-Date) -join ';'
        }
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    #
    [void]addstartlog($sampledb, $cmodule){
        #
        if ($cmodule -match 'vminform'){
            return
        }
        #
        if ($cmodule -match 'batch') {
            $mess = $this.project, $this.cohort,
                $this.batchid.PadLeft(2,'0'), ('START:' + $cmodule), (Get-Date) -join ';'
        } else {
            $mess = $this.project, $this.cohort,
                $this.slideid, ('START:' + $cmodule), (Get-Date) -join ';'
        }
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    #
    [void]adderrorlog($sampledb, $cmodule, $antibody){
        #
        $mess = $this.project, $this.cohort,
            $this.slideid, ('ERROR:' + $cmodule +
                ' - Antibody: ' + $antibody + ' - Algorithm:'), (Get-Date) -join ';'
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    #
    [void]addfinishlog($sampledb, $cmodule, $antibody){
        #
        $mess = $this.project, $this.cohort,
            $this.slideid, ('FINISH:' + $cmodule + ' - Antibody: ' + $antibody + ' - Algorithm:'), (Get-Date) -join ';'
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    #
    [void]addstartlog($sampledb, $cmodule, $antibody){
        #
        $mess = $this.project, $this.cohort,
            $this.slideid, ('START:' + $cmodule + ' - Antibody: ' + $antibody + ' - Algorithm:'), (Get-Date) -join ';'
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    #
}
#