<# -------------------------------------------
 testpswarpfits
 Benjamin Green, Andrew Jorquera
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the methods of warpfits are 
 functioning as intended
 -------------------------------------------#>
 Class testpswarpfits{
    #
    [string]$mpath 
    [string]$processloc
    [string]$basepath
    [string]$module = 'batchwarpfits'
    [string]$batchid = '8'
    [string]$project = '0'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$slideid = 'M21_1'
    [switch]$dryrun = $false
    [string]$pybatchflatfieldtest = 'melanoma_batches_3_5_6_7_8_9_v2'
    [string]$slidelist = '"L1_1|M148|M206|M21_1|M55_1|YZ71|ZW2|MA12"'
    #
    testpswarpfits(){
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_warpfits'))
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
        $this.launchtests()
    }
    #
    testpswarpfits($dryrun){
        #
        $this.mpath = '\\bki04\astropath_processing'
        $this.processloc = '\\bki08\e$'
        $this.basepath = '\\bki04\Clinical_Specimen'
        $this.batchid = '6'
        $this.project = '1'
        $this.slideid = 'M10_2'
        $this.dryrun = $true
        $this.launchtests()
        #
    }
    #
    launchtests(){
        Write-Host '---------------------test ps [warpfits]---------------------'
        $this.importmodule()
        $task = ($this.project, $this.batchid, $this.processloc, $this.mpath)
        $this.testpswarpfitsconstruction($task)
        $inp = batchwarpfits $task  
        $this.testprocessroot($inp)
        $this.testshreddatim($inp)
        $this.removewarpoctetsdep($inp)
        #
        $this.testwarpfitsinput($inp)
        $this.runpywarpfitsexpected($inp)
        $this.testlogsexpected($inp)
        #
        $inp.all = $true
        Write-Host 'test for all slides'
        $this.testwarpfitsinput($inp)
        $this.runpywarpfitsexpected($inp)
        $this.testlogsexpected($inp)
        #
        $inp.sample.finish(($this.module+'test'))
        Write-Host '.'
    }
    <# --------------------------------------------
    importmodule
    helper function to import the astropath module
    and define global variables
    --------------------------------------------#>
    importmodule(){
        Write-Host "."
        Write-Host 'importing module ....'
        Import-Module $this.apmodule
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
        $inp.sample.start(($this.module+'-test'))
        Write-Host '    command:'
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
    testpswarpfitsconstruction
    test that the meanimage object can be constucted
    --------------------------------------------#>
    [void]testpswarpfitsconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [warpfits] constructors started'
        #
        try {
            batchwarpfits  $task | Out-Null
        } catch {
            Throw ('[warpfits] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            meanimage  $task $log | Out-Null
        } catch {
            Throw ('[meanimage] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [warpfits] constructors finished'
        #
    }
    <#---------------------------------------------
    testprocessroot
    ---------------------------------------------#>
    [void]testprocessroot($inp){
        #
        Write-Host "."
        Write-Host 'test processing root started'
        Write-Host '    raw path:' $inp.processloc
        Write-Host '    data path:' $inp.processvars[0]
        Write-Host '    data.dat export path:' $inp.processvars[1]
        if (!(test-path $inp.processloc)){
            Throw ('did not make processloc: ' + $inp.processloc)
        }
        Write-Host 'test processing root finished'
        #
    }
    #
    [array]getmoduletask($inp){
        #
        $taskname = 'batchwarpfits'
        $inp.getmodulename()
        $dpath = $inp.processvars[0]
        $rpath = $inp.processvars[1]
        #
        $inp.getslideidregex('batchwarpfits')
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        $externallog = $inp.processlog($taskname)
        #
        return @($pythontask, $externallog)
        #
    }
    <#---------------------------------------------
    testshreddatim
    ---------------------------------------------#>
    [void] testshreddatim($inp){
        #
        Write-Host "."
        Write-Host 'test shred dat on images started'
        Write-Host '    get slide list from one slideid:' $this.slideid
        Write-Host '    open image keys text'
        $image_keys_file = $inp.getkeysloc()
        Write-Host '        keys file:' $image_keys_file
        $image_keys = $inp.sample.GetContent($image_keys_file)
        #
        $inp.getslideidregex('batchwarpfits')
        #
        Write-Host '    get keys for this file'
        $images= $inp.getslidekeypaths($inp.batchslides[0], $image_keys)
        Write-Host '    keys:'
        Write-Host '   '$images
        #
        if ($this.dryrun){
            $inp.shreddat($inp.batchslides[0], $images)
            Write-Host '    get dats for image files in keys'
            $inp.getwarpdats()
        }
        #
        write-host 'get warp dats finished'
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
    [void]testwarpfitsinput($inp){
        #
        if ($this.dryrun){
            return
        }
        #
        Write-Host "."
        Write-Host 'test for [batchwarpfits] expected input started' 
        #
        $flatwpath = '\\' + $inp.sample.project_data.fwpath
        $this.addwarpoctetsdep($inp)
        #
        if ($inp.all){
            $slides = $this.slidelist
            $wd = '--workingdir', ($this.mpath +
                    '\warping\Project_' + $this.project) -join ' '
        } else {
            $slides = '"M21_1"'
            $wd = '--workingdir', ($this.basepath + '\warping\Batch_'+
                    $this.batchid.PadLeft(2,'0')) -join ' '
        }
        #
        Write-Host '    collecting [warfits] defined task'
        $task = $this.getmoduletask($inp)
        #
        Write-Host '    collecting [user] defined task'
        $userpythontask = ('warpingcohort',
            $this.basepath, 
            '--shardedim3root', $flatwpath,
            '--sampleregex', $slides,
            '--flatfield-file', ($this.mpath + '\flatfield\flatfield_' +
                            $this.pybatchflatfieldtest + '.bin'),
            '--noGPU --no-log --ignore-dependencies --allow-local-edits',
            '--use-apiddef --project', $this.project.PadLeft(2,'0'), $wd
        ) -join ' '
        #
        Write-Host '[user] defined    :' [regex]::escape($userpythontask)'end' 
        Write-Host '[warpfits] defined:' [regex]::escape($task[0])'end' 
        #
        if (!([regex]::escape($userpythontask) -eq [regex]::escape($task[0]))){
            Throw ('user defined and [warpoctets] defined tasks do not match')
        }
        #
        $this.removewarpoctetsdep($inp)
        #
        Write-Host 'test for [batchwarpfits] expected input finished' 
        #
    }
    <#---------------------------------------------
    runpywarpfitsexpected
    ---------------------------------------------#>
    [void]runpywarpfitsexpected($inp){
        #
        Write-Host "."
        Write-Host 'test for [batchwarpfits] expected output slides started'
        Write-Host '    testing for all slides:' $inp.all
        #
        $task = $this.getmoduletask($inp)
        #
        if ($this.dryrun){
            Write-Host '    get keys for all slides'
            $inp.getwarpdats()
            $this.runpytesttask($inp, $task[0], $task[1])
        }  else {
            Write-Host '   '$task[0]
            Write-Host '   '$task[1]
        }
        #
        Write-Host 'test for [batchwarpfits] expected output slides finished'
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
        Write-Host 'test for [batchwarpfits] expected log output started'
        $inp.getmodulename()
        $inp.getslideidregex()
        $externallog = $inp.ProcessLog('batchwarpkeys') 
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
        Write-Host 'test for [batchwarpfits] expected log output finished'
        #
    }
 }
 #
 [testpswarpfits]::new() | Out-NUll
 exit 0


