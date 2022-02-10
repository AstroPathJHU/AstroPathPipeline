<# -------------------------------------------
 testpsmeanimage
 created by: Andrew Jorquera
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the methods of meanimage are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsmeanimage {
    #
    [string]$mpath 
    [string]$processloc
    [string]$basepath
    [string]$module = 'meanimage'
    [string]$slideid = 'M21_1'
    [string]$project = '0'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [switch]$fast_test = $true
    [string]$pytype = 'sample'
    #
    testpsmeanimage(){
        #
        $this.launchtests()
        #
    }
    #
    testpsmeanimage($project, $slideid){
        #
        $this.slideid = $slideid
        $this.project = $project
        $this.launchtests
        #
    }
    #
    [void]launchtests(){
        #
        Write-Host '---------------------test ps [meanimage]---------------------'
        $this.importmodule()
        $task = ($this.project, $this.slideid, $this.processloc, $this.mpath)
        # $this.testpsmeanimageconstruction($task)
        $inp = meanimage $task 
        # $this.testprocessroot($inp)
        # $this.testcleanupbase($inp)
        # $this.comparepymeanimageinput($inp)
        # $this.runpymeanimage($inp)
        $this.testlogs($inp)
        # $this.ReturnDataTest($inp)
        # $this.CleanupTest($inp)
        Write-Host '.'
    }
    #
    importmodule(){
        Import-Module $this.apmodule
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_meanimage'))
        $this.basepath = $this.uncpath(($PSScriptRoot + '\data'))
    }
    #
    [string]uncpath($str){
        $r = $str -replace( '/', '\')
        if ($r[0] -ne '\'){
            $root = ('\\' + $env:computername+'\'+$r) -replace ":", "$"
        } else{
            $root = $r -replace ":", "$"
        }
        return $root
    }
    #
    [void]testpsmeanimageconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [meanimage] constructors started'
        #
        $log = logger $this.mpath $this.module $this.slideid 
        #
        try {
            meanimage  $task | Out-Null
        } catch {
            Throw ('[meanimage] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            meanimage  $task $log | Out-Null
        } catch {
            Throw ('[meanimage] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [meanimage] constructors finished'
        #
    }
    #
    [void]testcleanupbase($inp){
        #
        Write-Host '.'
        Write-Host 'test cleanup base method started'
        #
        Write-Host '   copying old results to a safe location'
        $sor = $this.basepath, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        $des = $this.processloc, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        #
        Write-Host '   source:' $sor
        Write-Host '   destination:' $des
        $inp.sample.copy($sor, $des, '*')
        #
        Write-Host '   running cleanup protocol'
        $inp.cleanupbase()
        #
        if (test-path $sor){
            Throw 'meanimage directory still exists after cleanup'
        }
        #
        Write-Host '   results appear to be cleared replacing'
        #
        if ($this.fast_test){
            #
            Write-Host '   $fast_test: true... replacing masks'
            $inp.sample.copy($des, $sor, '*')
            $this.comparepaths($des, $sor, $inp)
            #
        }
        #
        # $inp.sample.removedir($des)
        #
        Write-Host 'test cleanup base method finished'

    }
    #
    [void]testprocessroot($inp){
        #
        Write-Host '.'
        Write-Host 'test processing dir preparation started'
        #
        $md_processloc = ($this.processloc, 'astropath_ws', $this.module, $this.slideid) -join '\'
        if (!([regex]::escape($md_processloc) -contains [regex]::escape($inp.processloc))){
            Write-Host 'meanimage module process location not defined correctly:'
            Write-Host $md_processloc '~='
            Throw ($inp.processloc)
        }
        #
        $inp.sample.CreateDirs($inp.processloc)
        #
        if (!(test-path $md_processloc)){
            Throw 'process working directory not created'
        }
        Write-Host 'test processing dir preparation finished'
        #
    }
    #
    [void]comparepymeanimageinput($inp){
        #
        Write-Host '.'
        Write-Host 'compare python [meanimage] expected input to actual started'
        #
        $md_processloc = ($this.processloc, 'astropath_ws',
             $this.module, $this.slideid, 'meanimage') -join '\'
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        [string]$userpythontask = (('meanimage', $this.pytype -join ''),
            $dpath, 
            $this.slideid, #'--sampleregex',
            '--shardedim3root', $rpath,
            ' --workingdir', $md_processloc,
            "--njobs '8'",
            '--allow-local-edits',
            '--skip-start-finish')
        #
        $inp.getmodulename()
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath)
        #
        if (!([regex]::escape($userpythontask) -eq [regex]::escape($pythontask))){
            Write-Host 'user defined and [meanimage] defined tasks do not match:'  -foregroundColor Red
            Write-Host 'user defined       :' [regex]::escape($userpythontask)'end'  -foregroundColor Red
            Write-Host '[meanimage] defined:' [regex]::escape($pythontask)'end' -foregroundColor Red
            Throw ('user defined and [meanimage] defined tasks do not match')
        }
        Write-Host 'python [meanimage] input matches -- finished'
        #
    }
    #
    [void]runpymeanimage($inp){
        Write-Host '.'
        Write-Host 'test python meanimage in workflow started'
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $inp.getmodulename()
        $pythontask = $inp.('getpythontask' + $inp.pytype)($dpath, $rpath) 
        $pythontask = $pythontask, '--selectrectangles' -join ' '
        #
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        #
        Write-Host '    meanimage command:'
        Write-Host '   '$pythontask  
        Write-Host '    external log:' $externallog
        Write-Host '    launching task'
        #
        $inp.sample.checkconda()
        etenv $inp.sample.pyenv()
        Invoke-Expression $pythontask *>> $externallog
        exenv
        #
        Write-Host 'test python meanimage in workflow finished'
        #
    }
    #
    [void]testlogs($inp){
        #
        Write-Host '.'
        Write-Host 'test python log output started'
        $inp.getmodulename()
        $externallog = $inp.ProcessLog($inp.pythonmodulename) 
        $inp.logoutput = $inp.sample.GetContent($externallog)
        Write-Host $inp.logoutput[0]
        Write-Host $inp.sample.project';'$inp.sample.cohort
        <#
        try {
            $inp.getexternallogs($externallog)
            Write-Host $inp.logoutput
        } catch {
            $err = $_.Exception.Message
                $expectedoutput = 'Python tasked launched but there was an ERROR.'
            if ($err -notcontains $expectedoutput){
                Write-Host $_.Exception.Message
            }
        }
        #>
        Write-Host 'test python log output finished'
        #
    }
    #
    [void]ReturnDataTest($inp){
        Write-Host '.'
        Write-Host 'Starting Return Data Test'
        #
        $sourcepath = $inp.processvars[0]
        $returnpath = $inp.sample.im3folder()
        Write-Host '    Source Path: ' $sourcepath '\meanimage'
        Write-Host '    Return Path: ' $returnpath
        #
        $inp.returndata()
        $presaved_results_path = $this.processloc, $this.slideid, 'im3\meanimage' -join '\'
        #
        if ($inp.processvars[4]) {
            #
            $this.comparepaths($presaved_results_path, $returnpath, $inp)
            #
        }
        #
        Write-Host 'Passed Return Data Test'
    }
    #
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
        $lista = Get-ChildItem $patha
        $listb = Get-ChildItem $pathb
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
    [void]CleanupTest($inp){
        Write-Host 'Starting Cleanup Test'
        if ($inp.processvars[4]) {
            $inp.cleanup()
            if (@(Test-Path $inp.processvars[0])) {
                Throw 'Cleanup Test Failed'
            }
        }
        Write-Host 'Processing Folder Deleted'
        Write-Host 'Passed Cleanup Test'
    }
}
#
# launch test and exit if no error found
#
[testpsmeanimage]::new() | Out-Null
exit 0