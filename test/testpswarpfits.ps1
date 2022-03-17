using module .\testtools.psm1
<# -------------------------------------------
 testpswarpfits
 Benjamin Green, Andrew Jorquera
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the methods of warpfits are 
 functioning as intended
 -------------------------------------------#>
 Class testpswarpfits : testtools {
    #
    [string]$module = 'batchwarpfits'
    [string]$class = 'batchwarpfits'

    #
    testpswarpfits() : base(){
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_warpfits'))
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
        $this.launchtests()
    }
    #
    testpswarpfits($dryrun) : base('1','M10_2','6', $dryrun){
        #
        $this.processloc = '\\bki08\e$'
        $this.basepath = '\\bki04\Clinical_Specimen'
        $this.launchtests()
        #
    }
    #
    launchtests(){
        #
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
        $this.compareinputs($userpythontask, $task[0])
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


