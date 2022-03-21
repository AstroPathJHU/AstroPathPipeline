  
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
    [string]$module = 'meanimage'
    #
    testpsqueue(){
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
        $this.launchtests()
    }
    #
    testpsqueue($dryrun){
        $this.slideid = "M1_1"
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
        $this.teststartmess($inp)
        $this.testchecktransfer($inp)
        $this.testcheckshredxml($inp)
        $this.testcheckmeanimage($inp)
        $this.testcheckbatchflatfield($inp)
        $this.testcheckwarpoctets($inp)
        $this.testcheckbatchwarpkeys($inp)
        $this.testbuildqueue($inp)
        $this.testextractqueue($inp)
        Write-Host '.'
    }
    #
    [void]importmodule(){
        Write-Host '.'
        Write-Host 'importing astropath ....'
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
        <#
        if ($log.vers -notmatch '0.0.1'){
            Throw 'version number wrong'
        }
        #>
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
        Write-Host '    check shredxml output:' $inp.checkshredxml($log, $false)
        Write-Host 'testing check shredxml fin'
    }
    #
    [void]testcheckmeanimage($inp){
        Write-Host '.'
        Write-Host 'testing check meanimage started'
        $log = logger $this.mpath 'meanimage' $this.slideid
        #
        $inp.updatelogger($log, 'meanimage')
        #
        Write-Host '    log version:' $log.vers 
        Write-Host '    check log output:' $inp.checklog($log, $true)
        #
        Write-Host '    check meanimage output:' $inp.checkmeanimage($log, $false)
        Write-Host 'testing check meanimage fin'
    }
    #
    [void]testcheckbatchflatfield($inp){
        Write-Host '.'
        Write-Host 'testing check batchflatfield started'
        $log = logger $this.mpath 'batchflatfield' $this.slideid
        #
        $inp.updatelogger($log, 'batchflatfield')
        #
        Write-Host '    log version:' $log.vers 
        #Write-Host '    check log output:' $inp.checklog($log, $true)
        #
        Write-Host '    check batchflatfield output:' $inp.checkbatchflatfield($log, $true)
        Write-Host 'testing check batchflatfield fin'
    }
    #
    [void]testcheckwarpoctets($inp){
        Write-Host '.'
        Write-Host 'testing check warpoctets started'
        $log = logger $this.mpath 'warpoctets' $this.slideid
        #
        $inp.updatelogger($log, 'warpoctets')
        #
        Write-Host '    log version:' $log.vers 
        Write-Host '    check log output:' $inp.checklog($log, $true)
        #
        Write-Host '    check warpoctets output:' $inp.checkwarpoctets($log, $false)
        Write-Host 'testing check warpoctets fin'
    }
    #
    [void]testcheckbatchwarpkeys($inp){
        Write-Host '.'
        Write-Host 'testing check batchwarpkeys started'
        $log = logger $this.mpath 'batchwarpkeys' $this.slideid
        #
        $inp.updatelogger($log, 'batchwarpkeys')
        #
        Write-Host '    log version:' $log.vers 
        Write-Host '    log message:' $log.slidelog
        Write-Host '    log message:' $log.mainlog
        Write-Host '    check log output:' $inp.checklog($log, $true)
        #
        Write-Host '    check batchwarpkeys output:' $inp.checkbatchwarpkeys($log, $false)
        Write-Host 'testing check batchwarpkeys fin'
    }
    #
    [void]testbuildqueue($inp){
        #
        Write-Host "."
        Write-Host 'test build queue start'
        #
        $slides = $inp.importslideids($this.mpath)
        #
        $projects = @('0')
        #
        $cleanedslides = $slides | 
            Where-Object {$projects -contains $_.Project}
        Write-Host '    checking:' 
        Write-Host ($cleanedslides | Format-Table | Out-String).Trim()
        $tasks = $inp.defNotCompletedSlides($cleanedslides)
        Write-Host '    tasks:' $tasks.slideid
        #
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
    [void]teststartmess($inp){
        #
        Write-Host '.'
        Write-Host 'test task started detection started'
        #
        $slides = $inp.importslideids($this.mpath)
        Write-Host '    create logger'
        $log = logger $this.mpath $this.module 'L1_1'
        #
        Write-Host '    "update" log'
        $log.Sample($this.slideid, $this.mpath, $slides)
        Write-Host '    "update" vers'
        $log.vers = $log.GetVersion($this.mpath, $this.module, $log.project, 'short')
        #
        #$log.start($this.module)
        #
        Write-Host '    slide log path:' $log.slidelog
        #
        Write-Host '    reading log'
        $loglines = $inp.opencsvfile($log.slidelog, ';',
             @('Project','Cohort','slideid','Message','Date'))
        #
        # parse log
        #
        $statustypes = @('START:','ERROR:','FINISH:')
        $savelog = @()
        $vers = $log.vers -replace 'v', ''
        $vers = ($vers -split '\.')[0,1,2] -join '.'
        #
        if ($log.slidelog -match [regex]::Escape($log.mainlog)){
            $ID= $log.BatchID
        } else {
            $ID = $log.slideid
        }
        #
        Write-Host '    log id:' $ID
        Write-Host '    log vers:' $vers
        #
        foreach ($statustype in $statustypes){
            $savelog += $loglines |
                    where-object {
                        ($_.Message -match $vers) -and 
                         ($_.Slideid -match $ID) -and 
                         ($_.Message -match $statustype)
                    } |
                    Select-Object -Last 1 
        }
        #
        $d1 = ($savelog | Where-Object {$_.Message -match $statustypes[0]}).Date
        $d2 = ($loglines |
                 Where-Object {
                    $_.Message -match $statustypes[1] -and
                     ($_.Slideid -match $ID)
                 }).Date |
               Select-Object -Last 1 
        $d3 = ($savelog | Where-Object {$_.Message -match $statustypes[2]}).Date
        #
        Write-Host '    start date:' $d1
        Write-Host '    error line:' $d2
        Write-Host '    fin date:' $d3
        #
        Write-Host '    test 1:' (!$d1)
        Write-Host '    test 2:' ($d1 -le $d2 -and $d3 -ge $d2) 
        Write-Host '    test 3:' (!$false -and ($d3 -gt $d1))
        Write-Host '    test 4:' ($false -and !($d3 -gt $d1))
        #
        Write-Host '    log status:' $inp.checklog($log, $false)
        #
        Write-Host 'test task started detection fin'
        #
    }
    #
}
#
# launch test and exit if no error found
#
try {
    [testpsqueue]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0

