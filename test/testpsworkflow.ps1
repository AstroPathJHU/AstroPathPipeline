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
        #
        $this.testastropathupdate($inp)
        $inp.workerloglocation = $PSScriptRoot + '\data\workflowlogs\'
        $inp.createdirs($inp.workerloglocation)
        $this.testworkerlistdef($inp)
        <#
        $this.testorphanjobmonitor($inp)
        $this.testwait($inp)
        #>
        $inp.removedir($PSScriptRoot + '\data\workflowlogs')
        #
        Write-Host '***running prepare sample'
        $inp.preparesample($this.slideid)
        Write-Host '***finished prepare sample'
        $this.savephenotypedata($inp)
        $this.removesetupvminform($inp)
        $this.testastropathupdate2('shredxml', $cred)
        $this.setupbatchwarpkeys($inp)
        #
        $this.testastropathupdate2('batchwarpkeys', $cred)
        $this.setupvminform($inp)
        $this.testastropathupdate2('vminform', $cred)
        $this.removesetupvminform($inp)
        $this.returnphenotypedata($inp)
        $this.cleanup($inp)
        $this.testcorrectionfile($inp, $true)
        $this.testgitstatus($inp)
        #
        # add user name and password to runtestjob(4) for these to work
        <#
        #$this.launchremotejobbatch($inp)  
        #$this.launchremotejob($inp) 
        #$this.launchremotetestjob($inp)  
        #>
        Write-Host '.'
        #
    }
    #
    [void]cleanup($sampledb){
        #
        write-host '.'
        write-host 'clean up started'
        $sampledb.removedir($this.mpath + '\across_project_queues')
        $sampledb.removedir($this.basepath + '\upkeep_and_progress\progress_tables')
        $sampledb.createnewdirs($this.basepath + '\logfiles')
        write-host 'clean up finished'
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
        wait-job $j -timeout 30
        Write-Host '    wait returned:' (Get-Date)
        #
        write-host '    job state:' $j.State
        #
        if(!($j.State -match 'Completed')){
            Write-Host '    Path exists:' (test-path $inp.workertasklog($jobname))
            $workertasklog = $inp.workertasklog($jobname)
            $workertasklog = $workertasklog -replace '\\', '/'
            $fileInfo = New-Object System.IO.FileInfo $workertasklog
            $fileStream = $fileInfo.Open([System.IO.FileMode]::Open)
            $fileStream.Dispose()
            Throw 'orphaned task monitor did not close correctly'
        }
        #
        Receive-Job $j -ErrorAction Stop 
        Receive-Job $testj -ErrorAction Stop 
        remove-job $testj
        remove-job $j
        #
        if (get-job){
            Throw (get-job.Name)
        }
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
                "&{Write-Host 'Launched'; write-host '-stringin:1-M21_1} 2'; Start-Sleep -s ($n); Write-Host 'Finished'}" *>> $workertasklog
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
        if (get-event) {
            Write-Host '    event trigger:' $true
        } else {
            Throw ' Error in detecting event after creation'
        }
        #
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
        if (!$a){
            Throw 'event did not trigger execution'
        }
        #
        Write-Host '    wait for the second event to trigger'
        $inp.waitany($j.id)
        #
        Write-Host '    second event triggered'
        if (get-event) {$a = $true} else {$a = $false}
        Write-Host '    event trigger:' $a
        Write-Host '    job state:' $j.State
        #
        if ($j.State -notmatch 'COMPLETED'){
            Throw 'job did not exit correctly'
        }
        #
        Receive-Job $j -ErrorAction Stop | Remove-Job
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
        $import_csv_file = $this.mpath + '\AstroPathAPIDdef.csv'
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
        Write-Host '    test slide confirmed'
        Write-Host '    remove the slideid test from the table'
        #
        $import_csv_file = $this.mpath + '\AstroPathAPIDdef.csv'
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
    [void]testastropathupdate2($module, $cred){
        #
        Write-Host '.'
        Write-Host 'test that astropath files update with the file watchers appropriately started'
        #
        write-host '    create new log'
        if ($module -match 'batch'){
            $log = logger -mpath 'v:\astropath_processing' `
                -module $module -project $this.project -batchid $this.batchid
        } else {
            $log = logger -mpath 'v:\astropath_processing' -module $module -slideid $this.slideid
        }
        #
        if ($module -match 'vminform'){
            $log.val = @{antibody = $this.informantibody; 
                algorithm = $this.informproject; informvers = $this.informvers}
        }
        #
        $log.removefile($log.mainlog)
        #
        write-host '    create astropath db'
        $workflow = astropathworkflow -MPATH 'V:\ASTROPATH_PROCESSING' -TEST -Credential $cred
        $workflow.buildsampledb()
        write-host '    update as a running task'
        #
        $log.start($module)
        $workflow.WaitAny()
        #
        if ($module -match 'vminform'){
            $status = ($workflow.moduleobjs.($module).localqueue.($workflow.project) |
                Where-Object {$_.slideid -match 'M21_1'}).($this.informantibody + '_Status')
            $this.addinformbatchlogs($workflow)
        } else {
            $status = ($workflow.moduleobjs.($module).maincsv |
                Where-Object {$_.slideid -match 'M21_1'}).Status
        }
        #
        if ($status -notmatch 'RUNNING'){
            Throw ('status after module launch not RUNNING: ' + $status)
        }
        write-host '    update as a finished task'
        #
        $this.addproccessedalgorithms($workflow)
        $log.finish($module)
        $workflow.WaitAny()
        #
        if ($module -match 'vminform'){
            $status = ($workflow.moduleobjs.($module).localqueue.($workflow.project) |
                Where-Object {$_.slideid -match 'M21_1'}).($this.informantibody + '_Status')
        } else {
            $status = ($workflow.moduleobjs.($module).maincsv |
                Where-Object {$_.slideid -match 'M21_1'}).Status
        }
        #
        if ($status -notmatch 'FINISHED'){
            Throw ('status after module launch not FINISHED: ' + $status)
        }
        #
        Write-Host 'test that astropath files update with the file watchers appropriately finished'
        #
    }
    #
    [void]launchremotetestjob($inp){
        #
        write-host '.'
        write-host 'test launching a remote job started'
        #
        $module = 'shredxml'
        $currenttask = @('0', $this.slideid)
        $currentworker = [PSCustomObject]@{
            module = $module
            server = 'bki02'
            location = 'e$'
            status = 'IDLE'
        }
        #
        $jobname = $inp.defjobname($currentworker)
        if (test-path $inp.workertasklog($jobname)){
            remove-item $inp.workertasklog($jobname) -force -confirm:$false
        }
        #
        $currenttaskinput = $inp.addedargs($module, $currenttask, $currentworker)
        #
        $newtask =  (' Import-Module "', $inp.coderoot(), '"
            $output = & {LaunchModule -mpath "', 
            $this.uncpath($inp.mpath), $currenttaskinput, '" -test} 2>&1
            $output.sample.popfile("',$inp.workerlogfile($jobname),'", $output.sample.vers)
            ') -join ''
        #
        set-content $inp.workertaskfile($jobname) $newtask
        #
        $this.runtestjob($inp, $jobname, $currentworker, $currenttask)
        $this.checkjob($inp, $jobname)
        #
    }
    #
    [void]launchremotejob($inp){
        #
        write-host '.'
        write-host 'test launching a remote job started'
        #
        $inp.mpath = '\\bki04\astropath_processing'
        #
        $module = 'imagecorrection'
        $currenttask = @('16', 'AP0160021')
        $currentworker = [PSCustomObject]@{
            module = $module
            server = 'bki09'
            location = 'g$'
            status = 'IDLE'
        }
        #
        $jobname = $inp.defjobname($currentworker)
        if (test-path $inp.workertasklog($jobname)){
            remove-item $inp.workertasklog($jobname) -force -confirm:$false
        }
        #
        $currenttaskinput = $inp.addedargs($module, $currenttask, $currentworker)
        Write-host '   '$currenttaskinput
        #
        $inp.buildtaskfile($jobname, $currenttaskinput)
        $this.runtestjob($inp, $jobname, $currentworker, $currenttask)
        $this.checkjob($inp, $jobname)
        #
        write-host 'test launching a remote job finished'
    }
    #
    [void]runtestjob($inp, $jobname, $currentworker, $currenttask){
        #
        $password = ConvertTo-SecureString "password" -AsPlainText -Force
        $cred = New-Object System.Management.Automation.PSCredential ("username", $password)  
        #
        $inp.login = $cred
        #
        $mtask = get-content $inp.workertaskfile($jobname)
        write-host '   '$mtask
        #
        $securestrings = $inp.Getlogin()       
        $currentworkerip = $inp.defcurrentworkerip($currentworker)
        #
        $inp.executetask($currenttask, $currentworker,
            $securestrings, $currentworkerip, $jobname)
    }
    #
    [void]launchremotejobbatch($inp){
        #
        write-host '.'
        write-host 'test launching a remote job started'
        #
        $inp.mpath = '\\bki04\astropath_processing'
        #
        $module = 'batchwarpkeys'
        $currenttask = @('16', '3')
        $currentworker = [PSCustomObject]@{
            module = $module
            server = 'bki06'
            location = 'n$'
            status = 'IDLE'
        }
        #
        $jobname = $inp.defjobname($currentworker)
        if (test-path $inp.workertasklog($jobname)){
            remove-item $inp.workertasklog($jobname) -force -confirm:$false
        }
        #
        $currenttaskinput = $inp.addedargs($module, $currenttask, $currentworker)
        Write-host '   '$currenttaskinput
        #
        $inp.buildtaskfile($jobname, $currenttaskinput)
        $this.runtestjob($inp, $jobname, $currentworker, $currenttask)
        $this.checkjob($inp, $jobname)
        #
        write-host 'test launching a remote job finished'
    }
    #
    [void]checkjob($inp, $jobname){
        #
        if (get-job){
            $j = get-job
            Write-host '    job launched'
            Write-host '    job state:' $j.state
            wait-job $j
            Write-host '    job state:' $j.state
            Receive-Job $j -ErrorAction Stop 
        }
        #
        if (test-path $inp.workertasklog($jobname)){
            #
            $file = get-content $inp.workertasklog($jobname)
            write-host '    worker log:' $file
            #
        }
        #
    }
}
#
# launch test and exit if no error found
#
try {
    [testpsworkflow]::new() | Out-Null
} catch {
    Throw $_.Exception
}
exit 0
