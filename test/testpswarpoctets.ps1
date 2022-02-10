<# -------------------------------------------
 testpswarpoctets
 created by: Andrew Jorquera
 Last Edit: 02.10.2022
 --------------------------------------------
 Description
 test if the methods of warpoctets are 
 functioning as intended
 -------------------------------------------#>
#
Class testpswarpoctets {
    #
    [string]$mpath 
    [string]$processloc
    [string]$basepath
    [string]$module = 'warpoctets'
    [string]$slideid = 'M21_1'
    [string]$project = '0'
    [string]$batchid = '8'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$batchbinfile
    [switch]$fast_test = $true
    #
    testpswarpoctets(){
        #
        $this.launchtests()
        #
    }
    testpswarpoctets($project, $slideid){
        #
        $this.slideid = $slideid
        $this.project = $project
        $this.launchtests
        #
    }
    #
    [void]launchtests(){
        #
        Write-Host '---------------------test ps [warpoctets]---------------------'
        $this.importmodule()
        $task = ($this.project, $this.slideid, $this.processloc, $this.mpath)
        $this.testpswarpoctetsconstruction($task)
        $inp = warpoctets $task
        #$this.testprocessroot($inp)
        #$this.testcleanupbase($inp)
        $this.comparepywarpoctetsinput($inp)
        #$this.runpywarpoctets($inp)
        # $this.CleanupTest($inp)
        Write-Host '.'
    }
    #
    importmodule(){
        Import-Module $this.apmodule
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.batchbinfile = $this.mpath + '\flatfield\flatfield_' + $this.batchid + '.bin'
        $this.processloc = $this.uncpath(($PSScriptRoot + '\test_for_jenkins\testing_warpoctets'))
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
    [void]testpswarpoctetsconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [warpoctets] constructors started'
        #
        $log = logger $this.mpath $this.module $this.slideid 
        #
        try {
            warpoctets  $task | Out-Null
        } catch {
            Throw ('[warpoctets] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try {
            warpoctets  $task $log | Out-Null
        } catch {
            Throw ('[warpoctets] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        Write-Host 'test [warpoctets] constructors finished'
        #
    }
    
    #
    [void]comparepywarpoctetsinput($inp){
        #
        Write-Host '.'
        Write-Host 'compare python [warpoctets] expected input to actual started'
        #
        $md_processloc = ($this.processloc, 'astropath_ws', $this.module, $this.slideid, 'warpoctets') -join '\'
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        [string]$userpythontask = ('warpingcohort',
            $dpath,
            '--shardedim3root', $rpath,
            '--sampleregex', $this.slideid,
            '--flatfield-file',  $this.batchbinfile,
            '--octets-only',
            '--noGPU',
            '--allow-local-edits',
            '--skip-start-finish',
            '--use-apiddef', 
            '--project', ($this.project).padleft(2,'0')-join ' ')
        #
        $pythontask = $inp.getpythontask($dpath, $rpath)
        if (!([regex]::escape($userpythontask) -eq [regex]::escape($pythontask))){
            Write-Host 'user defined and [warpoctets] defined tasks do not match:'  -foregroundColor Red
            Write-Host 'user defined       :' [regex]::escape($userpythontask)'end'  -foregroundColor Red
            Write-Host '[warpoctets] defined:' [regex]::escape($pythontask)'end' -foregroundColor Red
            Throw ('user defined and [warpoctets] defined tasks do not match')
        }
        Write-Host 'python [warpoctets] input matches -- finished'
        #
    }
    #
    [void]runpywarpoctets($inp){
        Write-Host '.'
        Write-Host 'test python warpoctets in workflow started'
        $rpath = $PSScriptRoot + '\data\raw'
        $dpath = $this.basepath
        $pythontask = $inp.getpythontask($dpath, $rpath)
        #
        $externallog = $inp.ProcessLog('warpingcohort') 
        #
        Write-Host '    warpoctets command:'
        Write-Host '   '$pythontask  
        Write-Host '    external log:' $externallog
        Write-Host '    launching task'
        #
        $inp.sample.checkconda()
        etenv $inp.sample.pyenv()
        Invoke-Expression $pythontask *>> $externallog
        exenv
        #
        try {
            $inp.getexternallogs($externallog)
        } catch {
            $err = $_.Exception.Message
             $expectedoutput = 'Python tasked launched but there was an ERROR.'
            if ($err -notcontains $expectedoutput){
                Write-Host $_.Exception.Message
            }
        }
        #
        Write-Host 'test python warpoctets in workflow finished'
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
    #
}
#
# launch test and exit if no error found
#
$test = [testpswarpoctets]::new()
exit 0