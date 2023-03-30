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
        $this.testpswarpfitsconstruction($this.task)
        $inp = batchwarpfits $this.task
        $inp.sample.teststatus = $true  
        $this.testprocessroot($inp)
        $this.testshreddatim($inp)
        $this.testwarpfitsinput($inp)
        $this.removewarpoctetsdep($inp)
        if (!$this.dryrun){
            $this.buildtestflatfield($inp)
            $this.runpytaskpyerror($inp)
            $this.testlogpyerror($inp)
            $this.runpytaskaperror($inp)
            $this.testlogaperror($inp)
            $this.setupsample($inp)
        }
        #
        $this.runpywarpfitsexpected($inp)
        $this.testlogsexpected($inp)
        #
        if (!$this.dryrun){
            $inp.all = $true
            $this.removewarpoctetsdep($inp)
            Write-Host 'test for all slides'
            $this.testwarpfitsinput($inp)
            $this.runpywarpfitsexpected($inp)
            $this.runpytaskpyerror($inp)
            $this.testlogpyerror($inp)
            $this.runpytaskaperror($inp)
            $this.testlogaperror($inp)
        }
        #
        $this.testwarpfitsinput($inp)
        $this.runpywarpfitsexpected($inp)
        $this.testlogsexpected($inp)
        #
        $this.cleanuptest($inp)
        #
        $inp.sample.finish(($this.module+'test'))
        $this.testgitstatus($inp.sample)
        Write-Host '.'
    }
    <# --------------------------------------------
    testpswarpfitsconstruction
    test that the meanimage object can be constucted
    --------------------------------------------#>
    [void]testpswarpfitsconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [batchwarpfits] constructors started'
        #
        try {
            batchwarpfits  $task | Out-Null
        } catch {
            Throw ('[batchwarpfits] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            meanimage  $task $log | Out-Null
        } catch {
            Throw ('[meanimage] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [batchwarpfits] constructors finished'
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
    [void]testshreddatim($inp){
        #
        Write-Host "."
        Write-Host 'test shred dat on images started'
        Write-Host '    get slide list from one slideid:' $this.slideid
        Write-Host '    open image keys text'
        $image_keys_file = $inp.getkeysloc()
        Write-Host '        keys file:' $image_keys_file
        if ($inp.all){
            $userdef = $this.basepath, 'warping', ('Project'+$this.project),
                'octets', 'image_keys_needed.txt' -join '\'
        } else {
            $userdef = $this.basepath, 'warping', ('Batch_'+$this.batchid.PadLeft(2,'0')),
                'octets', 'image_keys_needed.txt' -join '\'
        }
        #
        Write-Host '    [batchwarpfits] image keys file:' $image_keys_file
        Write-Host '    user defined:' $userdef
        #
        if ($userdef -notmatch [regex]::escape($image_keys_file)){
            Throw 'image key file not defined correctly'
        }
        #
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
        $flatwpath = $this.processloc, 'astropath_ws', 'batchwarpfits', $this.batchid -join '\'
        $this.addwarpoctetsdep($inp)
        #
        if ($inp.all){
            $slides = $this.slidelist
            $wd = '--workingdir', ($this.mpath +
                    '\warping\Project_' + $this.project) -join ' '
        } else {
            $slides = '"^(M21_1)$"'
            $wd = '--workingdir', ($this.basepath + '\warping\Batch_'+
                    $this.batchid.PadLeft(2,'0')) -join ' '
        }
        #
        Write-Host '    collecting [warfits] defined task'
        #
        $inp.getslideidregex($this.class)
        $inp.getmodulename()
        $dpath = $inp.processvars[0]
        $rpath = $inp.processvars[1]
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        #
        Write-Host '    collecting [user] defined task'
        $userpythontask = ('warpingmulticohort',
            $this.basepath, 
            '--shardedim3root', $flatwpath,
            '--sampleregex', $slides,
            '--flatfield-file', ($this.mpath + '\flatfield\flatfield_' +
                            $this.pybatchflatfieldtest + '.bin'),
            $inp.gpuopt(), '--no-log --ignore-dependencies --allow-local-edits',
            '--use-apiddef --job-lock-timeout 0:5:0', $wd
        ) -join ' '
        #
        $this.removewarpoctetsdep($inp)
        $this.compareinputs($userpythontask, $pythontask)
        #
    }
    <# --------------------------------------------
    runpytaskaperror
    check that the python task completes correctly 
    when run with the input that will throw a
    warpoctets sample error
    --------------------------------------------#>
    [void]runpytaskaperror($inp){
        #
        Write-Host '.'
        Write-Host 'test python [batchwarpfits] with error in processing started'
        Write-Host '    testing for all slides:' $inp.all
        #
        $this.addwarpoctetsdep($inp)
        $task = $this.getmoduletask($inp)
        #
        $inp.sample.CreateNewDirs($inp.processloc)
        #
        $pythontask = ($task[0] -replace `
            [regex]::escape($inp.sample.pybatchflatfieldfullpath()), 
            $this.batchreferencefile)
        #
        $externallog = $task[1] + '.err.log'
        $this.runpytesttask($inp, $pythontask, $externallog)
        $this.removewarpoctetsdep($inp)
        #
        Write-Host 'test python [batchwarpfits] with error in processing finished'
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
        $inp.sample.CreateNewDirs($inp.processloc)
        $this.addwarpoctetsdep($inp)
        $task = $this.getmoduletask($inp)
        #
        if ($this.dryrun){
            Write-Host '    get keys for all slides'
            $inp.getwarpdats()
            $this.runpytesttask($inp, $task[0], $task[1])
        }  else {
            #
            $this.addoctetpatterns($inp)
            $addedargs = (
                ' --initial-pattern-octets','2',
                '--principal-point-octets','2',
                '--final-pattern-octets','2'
            ) -join ' '
            #
            $pythontask = ($task[0] -replace `
                [regex]::escape($inp.sample.pybatchflatfieldfullpath()), 
                $this.batchreferencefile) + $addedargs
            $this.runpytesttask($inp, $pythontask, $task[1])
            #
        }
        #
        $this.removewarpoctetsdep($inp)
        Write-Host 'test for [batchwarpfits] expected output slides finished'
        #
    }
    #
    [void]cleanuptest($inp){
        #
        Write-Host '.'
        Write-Host 'test cleanup method started'
        #
        Write-Host '    remove working directories'
        Write-Host '    path expected to be removed:' ($this.basepath + '\warping')
        #
        if(!$this.dryrun){
            $inp.sample.removedir($this.basepath + '\warping')
        }
        #
        $inp.sample.removedir($this.processloc)
        #
        Write-Host '    running cleanup method'
        #
        $cleanuppath = (
            $this.processloc,
            'astropath_ws',
            $this.module,
            $this.slideid
        ) -join '\'
        #
        Write-Host '    path expected to be removed:' $cleanuppath
        $inp.cleanup()
        #
        if (Test-Path $cleanuppath) {
            Throw (
                'dir still exists -- cleanup test failed:', 
                $cleanuppath
            ) -join ' '
        }
        #
        $p2 = (
            $this.basepath,'\',
            $this.slideid,
            '\im3\meanimage\',
            $this.slideid,
            '-background_thresholds.csv'
        ) -join ''
        $inp.sample.removefile($p2)
        #
        Write-Host '    delete the testing_warpoctets folder'
        Write-Host '    path expected to be removed:' ($this.basepath + '\warping')
        $inp.sample.removedir(($this.basepath + '\warping'))
        #
        Write-Host '    delete the testing_warpoctets folder'
        Write-Host '    path expected to be removed:' ($this.mpath + '\warping')
        $inp.sample.removedir(($this.mpath + '\warping'))
        #
        Write-Host '    delete the testing_warpoctets folder'
        Write-Host '    path expected to be removed:' $this.processloc
        $inp.sample.removedir($this.processloc)
        #
    }
    #
 }
 #
 try {
    [testpswarpfits]::new() | Out-NUll
} catch {
    Throw $_.Exception.Message
}
exit 0


