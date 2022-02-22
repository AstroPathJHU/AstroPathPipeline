  
<# -------------------------------------------
 testpsqueue
 created by: Benjamin Green
 Last Edit: 1/10/2022
 --------------------------------------------
 Description
 test if the dispatcher works
 -------------------------------------------#>
#
Class testpsqueue {
    #
    [string]$mpath 
    [string]$process_loc
    [string]$slideid = "M21_1"
    [string]$module = 'shredxml'
    #
    testpsqueue(){
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
        $this.launchtests()
    }
    #
    testpsqueue($dryrun){
        $this.mpath = '\\bki04\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
        $this.launchtests()
    }
    #
    [void]launchtests(){
        Write-Host '---------------------test ps [queue]---------------------'
        $this.importmodule()
        #
        $inp = queue  $this.mpath $this.module
        #$this.testchecktransfer($inp)
        #$this.testcheckshredxml($inp)
        #$this.testbuildqueue($inp)
        $this.testextractqueue($inp)
        Write-Host '.'
    }
    #
    [void]importmodule(){
        Write-Host '.'
        Write-Host 'importing astropath ....'
        #$module = '\\bki08\e$\working_code\dev\AstroPathPipelinePrivate\astropath'
        $apmodule = $PSScriptRoot + '/../astropath'
        Import-Module $apmodule -EA SilentlyContinue
    }
    #
    [void]testchecktransfer($inp){
        Write-Host '.'
        Write-Host 'testing check transfer started'
        $log = logger $this.mpath 'shredxml' $this.slideid
        #
        $inp.updatelogger($log, 'transfer')
        #
        if ($log.vers -notmatch '0.0.1'){
            Throw 'version number wrong'
        }
        #
        Write-Host '    log version:' $log.vers 
        Write-Host '    check log output:' $inp.checklog($log, $true)
        #
        Write-Host '    check transfer output:' $inp.checktransfer($log)
        Write-Host 'testing check transfer fin'
    }
    #
    [void]testcheckshredxml($inp){
        Write-Host '.'
        Write-Host 'testing check shredxml started'
        $log = logger $this.mpath 'shredxml' $this.slideid
        #
        $inp.updatelogger($log, 'shredxml')
        #
        Write-Host '    log version:' $log.vers 
        Write-Host '    check log output:' $inp.checklog($log, $true)
        #
        Write-Host '    check transfer output:' $inp.checkshredxml($log, $false)
        Write-Host 'testing check shredxml fin'
    }
    #
    [void]testbuildqueue($inp){
        #
        Write-Host "."
        Write-Host 'test build queue start'
        #
        $slides = $inp.importslideids($this.mpath)
        Write-Host '   '($slides | Format-Table | Out-String).Trim()
        Write-Host 'test build queue fin'
        #
    }
    #
    [void]testextractqueue($inp){
        #
        Write-Host "."
        Write-Host 'extract queue start'
        #
        $inp.ExtractQueue()
        Write-Host '   '$inp.cleanedtasks
        #
        Write-Host 'extract queue fin'
        #
    }
    #
}
#
# launch test and exit if no error found
#
[testpsqueue]::new($true) | Out-Null
exit 0
