using module .\testtools.psm1
<# -------------------------------------------
 testpsimagecorrection
 Benjamin Green
 Last Edit: 2/1/2022
 --------------------------------------------
 Description
 test if the methods of imagecorrection are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsimagecorrection : testtools {
    #
    [string]$module = 'imagecorrection'
    [string]$class = 'imagecorrection'
    #
    testpsimagecorrection(){
        #
        $this.launchtests()
        #
    }
    #
    testpsimagecorrection($dryrun) : base ('1', 'M21_1', $dryrun){
        #
        $this.launchtests()
        #
    }
    #
    launchtests(){
        #
        $task = ($this.project, $this.slideid, $this.processloc, $this.mpath)
        $this.testpsimconstruction($task)
        $inp = imagecorrection $task
        <#
        $this.testrpath = $inp.processvars[1]
        #
        $this.testprocessroot($inp, $true)
        $inp.testpaths()
        $this.adddeps($inp)
        $this.testdownloadfiles($inp)
        $this.testshred($inp)
        $this.testimagecorrectioninput($inp)
        #
        $this.runpytaskexpected($inp)
        $this.testrenamefw2dat($inp)
        $this.testinjectdat($inp)
        $this.CleanupTest($inp)
        #>
        $inp.runimagecorrection()
        <#
        $this.removedeps($inp)
        $this.testgitstatus($inp.sample)
        Write-Host '.'
        #>
    }
    <# --------------------------------------------
    testpsimconstruction
    test that the meanimage object can be constucted
    --------------------------------------------#>
    [void]testpsimconstruction($task){
        #
        Write-Host "."
        Write-Host ('test ['+$this.class+'] constructors started')
        #
        #$log = logger $this.mpath $this.module -batchid:$this.batchid -project:$this.project 
        #
        try {
            imagecorrection  $task | Out-Null
        } catch {
            Throw ('['+$this.class +'] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            meanimage  $task $log | Out-Null
        } catch {
            Throw ('[meanimage] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host ('test ['+ $this.class +'] constructors finished')
        #
    }
    #
    [void]adddeps($inp){
        #
        if ($this.dryrun){
            return
        }
        #
        Write-host '.'
        write-host 'preparing dependencies started'
        $inp.sample.setfile($inp.sample.batchflatfield(), 'flatfield')
        $inp.sample.setfile($inp.sample.batchwarpingfile(), 'warping')
        write-host 'preparing dependencies finished'
        #
    }
    #
    [void]removedeps($inp){
        #
        if ($this.dryrun){
            return
        }
        #
        Write-host '.'
        write-host 'preparing dependencies started'
        $inp.sample.removefile($inp.sample.batchflatfield())
        $inp.sample.setfile($inp.sample.batchwarpingfile(), 'warping')
        write-host 'preparing dependencies finished'
        #
    }
    #
    [void]testdownloadfiles($inp){
        #
        Write-host '.'
        write-host 'test download files started'
        $inp.DownloadFiles()
        #
        $des = $inp.processvars[0] +'\'+
            $inp.sample.slideid+'\im3\'+$inp.sample.Scan()+,'\MSI'
        $sor = $inp.sample.im3folder()
        #
        #$this.comparepaths($des, $sor, $inp)
        #
        $des = $inp.processvars[1] +'\' + $inp.sample.slideid + '\'
        $sor = $inp.sample.xmlfolder()
        #
        #$this.comparepaths($des, $sor, $inp)
        #
    }
    #
    [void]testshred($inp){
        #
        if ($this.dryrun){
            $inp.ShredDat()
        } else{ 
            $this.createtestraw($inp)
        }
        #
    }
    #
    [void]testimagecorrectioninput($inp){
        #
        if ($this.dryrun){
            return
        }
        #
        Write-host '.'
        write-host 'test for [imagecorrection] expected input started'
        #
        Write-Host '    collecting [module] defined task'
        #
        $inp.getmodulename()
        $rpath = $this.testrpath
        $dpath = $this.basepath
        #
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath)
        #
        Write-Host '    collecting [user] defined tasks'
        #
        $wd = ' --workingdir', 
            ($this.processloc, 'astropath_ws\imagecorrection',
             $this.slideid -join '\')  -join ' '
        #
        $userpythontask = (('applyflatwcohort',
            $dpath,
            '--sampleregex', $this.slideid, 
            '--shardedim3root', $rpath,
            '--flatfield-file', ($this.mpath + '\flatfield\flatfield_' +
                $this.pybatchflatfieldtest + '.bin'), 
            '--warping-file', $this.pybatchwarpingfiletest, 
            "--njobs '8' --no-log --layers -1 --allow-local-edits",
            '--use-apiddef --project', $this.project.PadLeft(2,'0')
            ) -join ' ') + $wd
        #
        #$this.runpytesttask($inp, $userpythontask, $externaltask)
        $this.compareinputs($userpythontask, $pythontask)
        #
    }
    #
      <# --------------------------------------------
    runpytaskexpected
    check that the python task completes correctly 
    when run with the correct input.
    --------------------------------------------#>
    [void]runpytaskexpected($inp){
        #
        Write-Host '.'
        Write-Host 'test python [imagecorrection] in workflow started'
        $inp.sample.CreateDirs($inp.processloc)
        $rpath = $this.testrpath
        $dpath = $inp.sample.basepath
        $inp.getmodulename()
        #
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath) 
        #
        $pythontask = $pythontask
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        $this.runpytesttask($inp, $pythontask, $externallog)
        #
        Write-Host 'test python [imagecorrection] in workflow finished'
        #
    }
    #
    [void]testrenamefw2dat($inp){
        #
        if ($inp.sample.vers -match '0.0.1'){
            return
        }
        Write-host '.'
        Write-host 'rename fw 2 dat test started'
        write-host '    remove data.dat from:' ($inp.processvars[1] + '\' + $this.slideid)
        #
        $inp.renamefw2dat()
        #
        Write-host 'rename fw 2 dat test finished'
        #
    }
    #
    [void]testinjectdat($inp){
        #
        write-host '.'
        write-host 'test inject dat started'
        if ($this.dryrun){
            $inp.injectdat()
        } else{ 
            $this.createtestflatw($inp)
        }
        write-host 'test inject dat finished'
        #
    }
    #
    [void]CleanupTest($inp){
        Write-Host 'Starting Cleanup Test'
        $flatwim3path = $inp.sample.basepath + '\' + $inp.sample.slideid + '\im3'
        New-Item -Path $flatwim3path -Name "flatw" -ItemType "directory"
        $flatwim3path += '\flatw'
        Write-Host 'flatwim3path: ' $flatwim3path
        Write-Host 'Flatw IM3 folder: ' $inp.sample.flatwim3folder()
        if (!([regex]::Escape($inp.sample.flatwim3folder()) -contains [regex]::Escape($flatwim3path))){
            Throw ('MSI folder not correct: ' + $inp.sample.flatwim3folder() + '~=' + $flatwim3path)
        }
        Write-Host 'Recieved Correct FlatwIM3 Folder'
        #
        Write-Host 'Flatw Folder: ' $inp.sample.flatwfolder()
        #
        $inp.sample.removedir($this.processloc)
        #
        Write-Host 'Passed Cleanup Test'
    }
}
#
# launch test and exit if no error found
#
try { 
    [testpsimagecorrection]::new($dryrun) | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0