using module .\testtools.psm1
<# -------------------------------------------
 testpsbatchwarpkeys
 Benjamin Green, Andrew Jorquera
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the methods of batchwarpkeys are 
 functioning as intended
 -------------------------------------------#>
 Class testpsbatchwarpkeys : testtools{
    #
    [string]$module = 'batchwarpkeys'
    [string]$class = 'batchwarpkeys'
    #
    testpsbatchwarpkeys(): base(){
        $this.launchtests()
    }
    #
    testpsbatchwarpkeys($dryrun) : base('16', 'AP01600030', '2', $dryrun){
        #
        $this.processloc = '\\bki08\e$'
        $this.basepath = '\\bki-fs1\data01\Clinical_Specimen_14'
        $this.launchtests()
        #
    }
    #
    launchtests(){
        $this.testpsbatchwarpkeysconstruction($this.task)
        $inp = batchwarpkeys $this.task
        $inp.sample.teststatus = $true  
        $this.removewarpoctetsdep($inp)
        $this.testprocessroot($inp)
        $this.testwarpkeysinputbatch($inp)
        $this.runpywarpkeysexpected($inp)
        $this.testlogsexpected($inp)
        $this.testcleanup($inp)
        #
        $inp.all = $true# uses all slide from the cohort, 
        #   output goes to the mpath\warping\octets folder
        $inp.updateprocessloc()
        Write-Host 'test for all slides'
        $this.removewarpoctetsdep($inp)
        $this.testwarpkeysinputbatch($inp)
        $this.runpywarpkeysexpected($inp)
        $this.testlogsexpected($inp)
        $this.testcleanup($inp)
        $inp.sample.finish(($this.module+'-test'))
        #
        $this.testgitstatus($inp.sample)
        Write-Host '.'
    }
    <# --------------------------------------------
    testpswarpfitsconstruction
    test that the meanimage object can be constucted
    --------------------------------------------#>
    [void]testpsbatchwarpkeysconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [batchwarpkeys] constructors started'
        #
        #$log = logger $this.mpath $this.module -batchid:$this.batchid -project:$this.project 
        #
        try {
            batchwarpkeys  $task | Out-Null
        } catch {
            Throw ('[batchwarpkeys] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            meanimage  $task $log | Out-Null
        } catch {
            Throw ('[meanimage] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [batchwarpkeys] constructors finished'
        #
    }
    #
    [void]testprocessroot($inp){
        Write-Host '.'
        Write-host 'test creating sample dir started'
        Write-Host '    process location:' $inp.processloc
        #
        if ($inp.all){
            $testprocloc = $this.mpath + '\warping\Project_' + $this.project
        } else{
            $testprocloc = $this.basepath + '\warping\Batch_' + ($this.batchid).PadLeft(2,'0')
        }
        #
        if (!($inp.processloc -contains $testprocloc)){
            Write-Host '    test process loc:' $testprocloc
            Throw 'test process loc does not match [batchwarpkeys] loc'
        }
        Write-Host '    slide id:' $inp.sample.slideid
        Write-host 'test creating sample dir finished'
    }
    #
    [void]testwarpkeysinputbatch($inp){
        #
        if ($this.dryrun){
            return
        }
        #
        Write-Host "."
        Write-Host 'test for [batchwarpkeys] expected input started' 
        #
        $flatwpath = '\\' + $inp.sample.project_data.fwpath
        $this.addwarpoctetsdep($inp)
        #
        if ($inp.all){
            $slides = $this.slidelist
            $wd = ' --workingdir', ($this.mpath +
                 '\warping\Project_' + $this.project) -join ' '
        } else {
            $slides = '"^(M21_1)$"'
            $wd = ' --workingdir', ($this.basepath + '\warping\Batch_'+
                 $this.batchid.PadLeft(2,'0')) -join ' '
        }
        #
        Write-Host '    collecting [warpkeys] defined task'
        #
        $inp.getslideidregex($this.class)
        #
        $inp.updateprocessloc()
        $inp.getmodulename()
        $dpath = $inp.sample.basepath
        $rpath = '\\' + $inp.sample.project_data.fwpath
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        #
        Write-Host '    collecting [user] defined task'
        $userpythontask = (('warpingmulticohort',
            $this.basepath, 
            '--shardedim3root', $flatwpath,
            '--sampleregex', $slides,
            '--flatfield-file', ($this.mpath + '\flatfield\flatfield_' +
                         $this.pybatchflatfieldtest + '.bin'),
            '--octets-only',$inp.gpuopt(), '--no-log --ignore-dependencies --allow-local-edits',
            '--use-apiddef',
            '--job-lock-timeout 0:5:0'
        ) -join ' ') + $wd
        #
        $this.compareinputs($userpythontask, $pythontask)
        #
        $this.removewarpoctetsdep($inp)
        #
        Write-Host 'test for [batchwarpkeys] expected input finished' 
        #
    }
    #
    [void]runpywarpkeysexpected($inp){
        #
        Write-Host "."
        Write-Host 'test for [batchwarpkeys] expected output slides started'
        Write-Host '    testing for all slides:' $inp.all
        #
        $this.addwarpoctetsdep($inp)
        #
        $inp.getmodulename()
        write-host $inp.pythonmodulename
        write-host $inp.sample.pybatchflatfieldfullpath()
        write-host $inp.workingdir()
        write-host $inp.buildpyopts('cohort')
        $task = $this.getmoduletask($inp)
        $inp.getbatchwarpoctets()
        $this.runpytesttask($inp, $task[0], $task[1])
        #
        $this.removewarpoctetsdep($inp)
        #
        Write-Host 'test for [batchwarpkeys] expected output slides finished'
        #
    }
    #
    [void]testcleanup($inp){
        #
        if ($this.dryrun){
            return
        }
        #
        Write-Host '.'
        Write-Host 'test cleaning tasks up started'
        #
        if ($inp.all){
            $warpingkeysfolder = $this.mpath + '\warping\Project_' + $this.project       
        } else {
            $warpingkeysfolder =  ($this.basepath + 
                '\warping\Batch_'+ $this.batchid.PadLeft(2,'0'))
        }
        #
        Write-Host '    path expected to be removed:' $warpingkeysfolder
        #
        $inp.cleanup()
        $inp.sample.removedir($warpingkeysfolder)
        #
        if (test-path $warpingkeysfolder){
            Throw 'cleanup did not delete folder, path still exists'
        }
        #
        $inp.sample.removedir($this.mpath + '\warping')
        #
        Write-Host 'test cleaning tasks up finish'
        #
    }
    #
#
}
#
try {
    [testpsbatchwarpkeys]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0
