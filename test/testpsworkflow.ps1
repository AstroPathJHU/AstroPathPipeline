  
using module .\testtools.psm1
<# -------------------------------------------
 testpsdistpatcher
 created by: Benjamin Green
 Last Edit: 1/10/2022
 --------------------------------------------
 Description
 test if the dispatcher works
 -------------------------------------------#>
#
Class testpsworkflow : testtools {
    #
    [string]$class = 'workflow'
    #
    testpsworkflow(): base(){
        #
        $password = ConvertTo-SecureString "MyPlainTextPassword" -AsPlainText -Force
        $cred = New-Object System.Management.Automation.PSCredential ("username", $password)  
        #
        $this.testconstructors($cred) 
        $inp = astropathworkflow -Credential $cred -mpath $this.mpath -test
        $this.testastropathupdate($inp)
        $inp.workerloglocation = $PSScriptRoot + '\data\workflowlogs\'
        $inp.createdirs($inp.workerloglocation)
        $this.testworkerlistdef($inp)
        $this.testorphanjobmonitor($inp)
        $this.testwait($inp)
        $inp.removedir($inp.workerloglocation)
        Write-Host '.'
        #
    }
    #
    [void]testconstructors([PSCredential]$cred){
        #
        Write-Host '.'
        Write-Host '[astropathworkflow] construction tests started'
       <#
        try {
            astropathworkflow -Credential $cred -test | Out-NULL
        } catch {
            Throw ('[astropathworkflow] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        #>
        try {
            astropathworkflow -Credential $cred -mpath $this.mpath -test | Out-NULL
        } catch {
            Throw ('[astropathworkflow] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            astropathworkflow -Credential $cred -mpath $this.mpath -projects @('1', '2') -test | Out-NULL
        } catch {
            Throw ('[astropathworkflow] construction with [3] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            astropathworkflow -Credential $cred -mpath $this.mpath -projects @('1', '2') `
                -submodules @('transfer', 'shredxml') -test | Out-NULL
        } catch {
            Throw ('[astropathworkflow] construction with [4] input(s) failed. ' + $_.Exception.Message)
        }
        #
        Write-Host '[astropathworkflow] construction tests finished'
        #
    }
    #
    [void]testworkerlistdef($inp){
        Write-Host '.'
        Write-Host 'test that the worker list can be defined started'
        #
        Write-Host "    Defining worker list"
        Write-Host '    mpath:' $inp.mpath
        #
        $inp.importworkerlist($inp.mpath)
        Write-Host ($inp.worker_data | Format-table | Out-String)
        $inp.printworkerlist()
        Write-Host '    check for running tasks'
        $inp.CheckOrphan()
        #
        if ($inp.worker_data.Status -match 'RUNNING'){
            Throw 'Some workers tagged as running when they are not'
        }
        #
        Write-Host 'test that the worker list can be defined finished'

    }
    #
    [void]testorphanjobmonitor($inp){
        #
        Write-Host "."
        Write-Host 'test orphan job monitor started'
        #
        Write-Host '    create a test job'
        $inp.defworkerlist()
        $this.StartTestJob($inp)
        #
        Write-Host '    launch orphan monitor for test job'
        $inp.CheckOrphan()
        #
        $currentworker = $inp.worker_data[0]
        $jobname = $inp.defjobname($currentworker)
        $j = get-job -Name $jobname
        #
        Write-Host '    job name:' $jobname
        #
        if (!($j) -OR (!($inp.worker_data.Status -match 'RUNNING'))){
            Throw 'orphaned task monitor failed to launch'
        }
        #
        start-sleep -s (1*15)
        if ($j.State -match 'Completed'){
            Throw 'orphaned task monitor exited early'
        }
        #
        $testj = get-job -Name ($jobname + '-test')
        Write-Host '    job start' (Get-Date)
        Write-Host '    wait for job'
        wait-job $testj -timeout 180
        wait-job $j -timeout 180
        Write-Host '    wait returned:' (Get-Date)
        #
        write-host '    job state:' $j.State
        #
        if(!($j.State -match 'Completed')){
            Write-Host '    Path exists:' (test-path $inp.workertasklog($jobname))
            $fileInfo = New-Object System.IO.FileInfo $inp.workertasklog($jobname)
            $fileStream = $fileInfo.Open([System.IO.FileMode]::Open)
            $fileStream.Dispose()
            Throw 'orphaned task monitor did not close correctly'
        }
        #
        Receive-Job $j -ErrorAction Stop
        #
        Write-Host 'test orphan job monitor finished'
        #
    }
    #
    [void]StartTestJob($inp){
        Write-Host '    Starting test job'
        #
        $currentworker = $inp.worker_data[0]
        $jobname = $inp.defjobname($currentworker)
        #
        $this.launchjob(($jobname + '-test'), 60, $inp.workertasklog($jobname))
        #
        Write-Host '    Test job launched'
    }
    #
    [void]launchjob($jobname, $n, $log){
        #
        Write-Host '    job name:' $jobname
        Write-Host '    file:' $log
        #
        $sb = {
            param($workertasklog, $n)
            pwsh -noprofile -executionpolicy bypass -command `
                "&{Write-Host 'Launched'; Start-Sleep -s ($n); Write-Host 'Finished'}" *>> $workertasklog
        }
        #
        $myparameters = @{
            ScriptBlock = $sb
            ArgumentList = $log, $n
            name = $jobname
            }
        #
        Start-Job @myparameters
        #
        if ((get-job).Name -notcontains $jobname){
            Throw 'test job not launched'
        }
        #
        $j = get-job -Name $jobname
        Wait-Job $j -Timeout 5
        if ($j.State -notmatch 'Running'){
            Receive-Job $j -ErrorAction Stop
        }
        #
        if (!(test-path $log)){
            Throw ('log not created: ' + $log)
        }
        #
    }
    #
    [void]StartEventWatcher($inp, $filename){
        Write-Host '    Starting event watcher'
        #
        Write-Host '    filename:' $filename
        $inp.filewatcher($filename)
        #
        Write-Host '        writing message in file'
        $inp.setfile($filename, 'Event START')
        Write-Host '        detect intial creation event'
        #
        if (get-event) {$a = $true} else {$a = $false}
        Write-Host '    event trigger:' $a
        get-event | remove-event
        #
        Write-Host '    event watcher launched'
    }
    #
    [void]testwait($inp){
        #
        Write-Host '.'
        Write-Host 'test waiting for a job or task started'
        #
        Write-Host '    create a test file for the event test'
        #
        $inp.defworkerlist()
        $currentworker = $inp.worker_data[0]
        $jobname = $inp.defjobname($currentworker)
        $filename = $inp.workertasklog($jobname) + '-eventtest'
        #
        $this.StartEventWatcher($inp, $filename)
        $this.StartTestJob($inp)
        #
        # create a job that will edit a
        # file in x seconds
        #
        Write-Host '    create event signaling job'
        $this.launchjob(($jobname + '-eventtest'), 10, $filename)
        #
        Write-Host '    wait for the first event to trigger'
        $j = get-job -Name ($jobname + '-test')
        $inp.waitany($j.id)
        #
        Write-Host '    first event triggered'
        if (get-event) {$a = $true} else {$a = $false}
        Write-Host '    event trigger:' $a
        get-event | remove-event
        Write-Host '    job state:' $j.State
        #
        Write-Host '    wait for the second event to trigger'
        $inp.waitany($j.id)
        #
        Write-Host '    second event triggered'
        if (get-event) {$a = $true} else {$a = $false}
        Write-Host '    event trigger:' $a
        Write-Host '    job state:' $j.State
        #
        Receive-Job $j -ErrorAction Stop
        $inp.UnregisterEvent($filename)
        Write-Host 'test waiting for a job or task finished'
        #
    }
    #
    [void]testastropathupdate($inp){
        Write-Host '.'
        Write-Host 'test that astropath files update with the file watchers appropriately started'
        #
        Write-Host '    import all tables'
        $inp.importaptables($inp.mpath, $true)
        #
        Write-Host '    add a test slide to the astropathAPIDdef'
        $import_csv_file = $this.mpath + '\AstropathAPIDdef.csv'
        $project_data = $inp.OpencsvFileConfirm($import_csv_file)
        $newobj = [PSCustomObject]@{
            SlideID = 'test'
            SampleName = 'test'
            Project = '0'
            Cohort = '0'
            Scan = '1'
            BatchID = '1'
            isGood = 1
            layer_n = 1
            delta_over_sigma_std_dev = .95
        }
        #
        $project_data += $newobj
        #
        $project_data | Export-CSV $import_csv_file -NoTypeInformation
        #
        # update the slideids and check for the test slideid
        #
        Write-Host '    table before update'
        Write-Host ($inp.slide_data | format-table | out-string)
        Write-Host '    update the slideids in the astropath table'
        $inp.WaitAny()
        Write-Host ($inp.slide_data | format-table | out-string)
        #
        Write-Host '    check that the test slide is in the astropathtable'
        if (!($inp.slide_data.slideid -match 'test')){
            Throw 'slide id test is not in the slide variable'
        }
        #
        Write-HOst '    test slide confirmed'
        Write-Host '    remove the slideid test from the table'
        #
        $import_csv_file = $this.mpath + '\AstropathAPIDdef.csv'
        $project_data = $inp.OpencsvFileConfirm($import_csv_file)
        $project_data = $project_data | Where-Object {$_.SlideID -ne 'test'}
        $project_data | Export-CSV $import_csv_file -NoTypeInformation
        #
        Write-Host '    updating the slideids in the astropath table'
        $inp.waitany()
        #
        Write-Host '    check that the slideid has been removed from the astropath table'
        if ($inp.slide_data.slideid -match 'test'){
            Throw 'slide id test is still in the slide variable'
        }
        #
        Write-HOst '    test slide confirmed'       
        #
        Write-Host 'test that astropath files update with the file watchers appropriately finished'
        #

    }
    #

}
#
# launch test and exit if no error found
#
try {
    [testpsworkflow]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0
