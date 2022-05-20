#
class astropathwftools : sampledb {
    #
    [Pscredential]$login
    [string]$workerloglocation = '\\' + $env:ComputerName +
        '\c$\users\public\astropath\'   
    #
    astropathwftools(){
        $this.apwftoolsinit("NA", '')
    }
    #
    astropathwftools($login){
        $this.apwftoolsinit($login, '')
    }
    #
    astropathwftools($login, $mpath) : base($mpath) {
        $this.apwftoolsinit($login, '')
    }
    #
    astropathwftools($login, $mpath, $projects) : base($mpath, $projects){
        $this.apwftoolsinit($login, '')
    }
        #
    astropathwftools($login, $mpath, $projects, $modules) : base($mpath, $projects){
        $this.apwftoolsinit($login, $modules)
    }
    <# -----------------------------------------
    apwftoolsinit
    store the input 
    ------------------------------------------
    Usage: $this.apwftoolsinit()
   ----------------------------------------- #>  
    [void]apwftoolsinit([Pscredential]$login, $modules){
        #
        $this.login = $login
        if ($modules){
            $this.modules = $modules
        } 
        #
        $this.writeoutput(" Username: " + $this.login.UserName)
        #
    }
    <# -----------------------------------------
    defworkerlist
    Builds the worker list for all modules
     from the AstroPathHPFsWlocs file
    ------------------------------------------
    Usage: $this.defworkerlist()
   ----------------------------------------- #>   
   [void]defworkerlist(){
        #
        $this.writeoutput(" Finding Workers")
        $this.importworkerlist($this.mpath)
        $this.CheckOrphan()
        $this.printworkerlist()
        #
    }
    <# -----------------------------------------
    defworkerlist
    Builds the worker list for all modules
     from the AstroPathHPFsWlocs file
    ------------------------------------------
    Usage: $this.defworkerlist()
   ----------------------------------------- #>   
   [void]printworkerlist(){
        $this.writeoutput("    Worker Status:")
        write-host "    " ($this.worker_data | 
            Format-Table  -AutoSize @{Name="module";Expression = { $_.module }; Alignment="left" },
                            @{Name="server";Expression = { $_.server }; Alignment="left" },
                            @{Name="location";Expression = { $_.location }; Alignment="left" },
                            @{Name="state";Expression = { $_.State }; Alignment="left" },
                            @{Name="status";Expression = { $_.Status }; Alignment="left" }  |
            Out-String).Trim() -ForegroundColor Yellow
       $this.writeoutput(" .") 
   }
    <# -----------------------------------------
    CheckOrphan
    Check for orphaned tasks from a previously
    running instance of astropath. If it finds
    an orphaned task it will create a new job
    with the orphaned jobs name that peridoucally
    checks for access. when access is confirmed
    the job exists and allows the worker to 
    seemlessly jump back into the workflow.
    ------------------------------------------
    Usage: $this.CheckOrphan()
    ----------------------------------------- #>
    [void]CheckOrphan(){
        #
        $idleworkers = $this.worker_data | where-object {$_.status -match 'IDLE'}
        foreach ($worker in @(0..($idleworkers.Length - 1))){
            #
            $jobname = $this.defjobname($idleworkers[$worker])
            $workertasklog = $this.workertasklog($jobname)
            $workertaskfile = $this.workertaskfile($jobname)
            #
            if (test-path $workertasklog){
                #
                $fileInfo = New-Object System.IO.FileInfo $workertasklog
                #
                $this.writeoutput("     Orphaned job found: $workertasklog")
                #
                try {
                    #
                    # check if I can access and remove the worker task file
                    #
                    $fileStream = $fileInfo.Open([System.IO.FileMode]::Open)
                    $fileStream.Dispose()
                    $this.removefile($workertasklog)
                    $this.CheckTaskLog($jobname, 'ERROR')
                    $this.removefile($workertaskfile)
                    $idleworkers[$worker].Status = 'IDLE'
                    $this.writeoutput(
                        "     Orphaned job completed and cleared: $workertasklog"
                    )
                }catch {
                    #
                    $this.StartOrphanMonitor($jobname)
                    $idleworkers[$worker].Status = 'RUNNING'
                    $this.writeoutput(
                        "     Orphaned job not completed: $workertasklog ... created worker log watchdog"
                    )
                }
            }
        }
        #
    }
    <# -----------------------------------------
     StartOrphanMonitor
     Start an orphaned job monitor 
     ------------------------------------------
     Usage: $this.StartOrphanMonitor()
    ----------------------------------------- #>
    [void]StartOrphanMonitor($jobname){
        #
        # start a new job monitoring this file with the same job name 
        #
        $myscriptblock = {
            param($workertasklog)
            #
            $workertasklog = $workertasklog -replace '\\', '/'
            $fpath = Split-Path $workertasklog
            $fname = Split-Path $workertasklog -Leaf
            #
            $newwatcher = [System.IO.FileSystemWatcher]::new($fpath)
            $newwatcher.Filter = $fname
            $newwatcher.NotifyFilter = 'LastWrite'
            #
            $SI = ($workertasklog) 
            #
            Register-ObjectEvent $newwatcher `
                -EventName Changed `
                -SourceIdentifier $SI | Out-Null
            #
            while (1) {
                try { 
                    $fileInfo = New-Object System.IO.FileInfo $workertasklog
                    $fileStream = $fileInfo.Open([System.IO.FileMode]::Open)
                    $fileStream.Dispose()
                    break
                } catch {
                    #
                    wait-event $SI  
                    remove-event -SourceIdentifier $SI
                    Start-Sleep -s 10
                    #
                }
                #
            }
            #
            Unregister-Event -SourceIdentifier $SI -Force 
            #
        }
        #
        $myparameters = @{
            ScriptBlock = $myscriptblock
            ArgumentList = $this.workertasklog($jobname)
            name = $jobname
        }
        Start-Job @myparameters
        #
    }
    <# -----------------------------------------
     distributetasks
     distribute tasks to active workers for 
     each module
     ------------------------------------------
     Usage: $this.distributetasks()
    ----------------------------------------- #>
    [void]distributetasks(){
        #
        foreach ($module in $this.modules){
            $currentworkers = $this.worker_data  | 
                where-object {
                    $_.module -match $module -and 
                    $_.status -match 'IDLE' -and 
                    $_.state -match $this.onstrings
                }
            #
            $cqueue = $this.moduletaskqueue.($module) 
            #
            if ($cqueue.count -ne 0 -and $currentworkers){
                $this.writeoutput(" Starting task distribution for $module")
                if ($module -match 'vminform'){
                    $this.sortvmqqueue($cqueue)
                    $cqueue = $this.moduletaskqueue.($module) 
                }
            }
            #
            while ($cqueue.count -ne 0 -and $currentworkers){
                #
                $currenttask = $cqueue.dequeue()
                $currentworker, $currentworkers = $currentworkers
                #
                $this.writeoutput(("     Launching Task on:", 
                    $currentworker.server, $currentworker.location -join ' '))
                $this.writeoutput("     $currenttask")
                #
                $this.launchtask($currenttask, $currentworker)
                #
                $currentworker.status = 'RUNNING'
                #
            }
            #
        }
        #
    }
    #
    [void]sortvmqqueue($cqueue){
        #
        $taskids = $cqueue | foreach-object {$_[0]}
        $this.moduletaskqueue.vminform = New-Object System.Collections.Generic.Queue[array]
        #
        $this.vmq.maincsv | 
            Where-Object {$_.taskid -in $taskids} |
            foreach-object {
                $this.moduletaskqueue.vminform.enqueue(@($_.taskid, $_.slideid))
            }
        #
    }
    <# -----------------------------------------
     LaunchTask
     Launch a task on a current worker
     ------------------------------------------
     Usage: $this.LaunchTask($currenttask, $currentworker)
     ------------------------------------------
     Input: 
        $currenttask: the task array inputs. for a:
            slide task: @(project, slideid)
            batch task: @(project, batchid)
            vminform task: @(project, slideid, antibody, algorithm)
        $[system.object]currentworker: four part array
            of a task object with fields:
                module, server, location, status
    ----------------------------------------- #>
    [void]LaunchTask($currenttask, $currentworker){
        #
        $securestrings = $this.Getlogin()       
        $currentworkerip = $this.defcurrentworkerip($currentworker)
        $jobname = $this.defjobname($currentworker)
        $currenttask = $this.getvminformtask($currenttask, $currentworker)
        $this.PrepareWorkerFiles(
            $currentworker.module, $currenttask, $jobname, $currentworker)
        $this.executetask($currenttask, $currentworker,
            $securestrings, $currentworkerip, $jobname)
        #
    }
    <# -----------------------------------------
     getvminformtask
     get the current info for the vminform task
     and update the main queue for launching the
     task
     ------------------------------------------
     Usage: $this.getvminformtask($currenttask, $currentworker)
    ----------------------------------------- #>
    [array]getvminformtask($currenttask, $currentworker){
        #
        if ($currentworker.module -notmatch 'vminform'){
            return $currenttask
        }
        #
        $row = $this.vmq.maincsv | Where-Object {
            $_.taskid -contains $currenttask[0]
        }
        #
        $row.ProcessingLocation = 'Processing:' + $currentworker.location
        $row.StartDate = Get-Date
        #
        $this.vmq.writemainqueue()
        #
        $currenttask = @($row.taskid, $row.slideid, $row.antibody, $row.algorithm)
        #
        return $currenttask
    }
    <# -----------------------------------------
     PrepareWorkerFiles
     set up the worker task file locations and 
     prepare the unique task input string to
     launchmodule.
     ------------------------------------------
     Usage: $this.PrepareWorkerFiles($currenttask, jobname, $currentworker)
    ----------------------------------------- #>
    [void]PrepareWorkerFiles($module, $currenttask, $jobname, $currentworker){
        #
        $this.createdirs($this.workerloglocation)
        $currenttaskinput = $this.addedargs($module, $currenttask, $currentworker)
        $this.buildtaskfile($jobname, $currenttaskinput)
        #
    }
    <# -----------------------------------------
    addedargs
    add the current task input for the various 
    module types: vminform, batch, or slide
    ----------------------------------------- #>
    [string]addedargs($cmodule, $currenttask, $currentworker){
        #
        $currenttaskinput = ''
        #
        switch -regex ($cmodule) {
            'vminform'{
                $currenttaskinput = '" -module ', $cmodule,
                    ' -slideid ', $currenttask[1],
                    ' -antibody ', $currenttask[2], ' -algorithm ', $currenttask[3],
                    ' -informvers ', $this.vmq.informvers -join '"'
            } 
            'batch' {
                $currentworkerstring = '\', $currentworker.server,
                    $currentworker.location -join '\'
                $currenttaskinput = '" -module ', $cmodule,
                    ' -project ', $currenttask[0], ' -batchid ', $currenttask[1],
                    ' -processloc ', $currentworkerstring -join '"'
            }
            default {
                $currentworkerstring = '\', $currentworker.server,
                    $currentworker.location -join '\'
                $currenttaskinput = '" -module ', $cmodule, 
                    ' -project ', $currenttask[0], ' -slideid ', $currenttask[1],
                    ' -processloc ', $currentworkerstring -join '"'
            }
        }
        #
        return $currenttaskinput
        #
    }
    <# -----------------------------------------
     buildtaskfile
     create the string to be added to the task
     file.
     ------------------------------------------
     Usage: $this.buildtaskfile($jobname, currenttaskinput)
    ----------------------------------------- #>
    [void]buildtaskfile($jobname, $currenttaskinput){
        #
        $currenttasktowrite = (' Import-Module "', $this.coderoot(), '"
        $output.output = & {LaunchModule -mpath:"',
                $this.mpath, $currenttaskinput,'"} 2>&1
            if ($output.output -ne 0){ 
                #
                $count = 1
                #
                $output.output | Foreach-object {
                    $output.popfile("',
                        $this.workerlogfile($jobname),
                        '", ("ERROR: " + $count + "`r`n"))
                    $output.popfile("',
                        $this.workerlogfile($jobname),
                        '", ("  " + $_.Exception.Message  + "`r`n"))
                    $s = $_.ScriptStackTrace.replace("at", "`t at")
                    $output.popfile("',
                        $this.workerlogfile($jobname),'", ($s + "`r`n"))
                    $count += 1
                }
                #
            } else {
                $output.popfile("',
                    $this.workerlogfile($jobname),
                    '", "Completed Successfully `r`n")
            }') -join ''
        #
        $this.SetFile($this.workertaskfile($jobname), $currenttasktowrite)
        #
    }
    #
    [void]executetask($currenttask, $currentworker,
        $securestrings, $currentworkerip, $jobname){
        #
        if ($currentworker.location -match 'VM'){
            $myscriptblock = {
                param($username, $securestring, $currentworkerip, $workertaskfile)
                psexec -i -nobanner -accepteula -u $username -p $securestring \\$currentworkerip `
                    pwsh -noprofile -executionpolicy bypass -command "$workertaskfile" `
                    *>> ($workertaskfile -replace '.ps1', '-job.log')
            }
        } else {
            $myscriptblock = {
                param($username, $securestring, $currentworkerip, $workertaskfile)
                psexec -nobanner -accepteula -u $username -p $securestring \\$currentworkerip `
                    pwsh -noprofile -WindowStyle Hidden -executionpolicy bypass -command "$workertaskfile" `
                    *>> ($workertaskfile -replace '.ps1', '-job.log')
            }
        }
        #
        $myparameters = @{
            ScriptBlock = $myscriptblock
            ArgumentList = $securestrings[0], $securestrings[1],
             $currentworkerip, $this.workertaskfile($jobname)
            name = $jobname
            }
        #
        Start-Job @myparameters
        $taskid = ($jobname, (Get-Date)) -join '-'
        [string]$logline = @("START: ", $taskid,"`r`n") -join ''
        $this.popfile($this.workerlogfile($jobname), $logline)
        #
    }
   <# -----------------------------------------
     WaitAny
     after jobs are launched from the queue 
     wait for events or jobs
     ------------------------------------------
     Usage: $this.WaitAny()
    ----------------------------------------- #>
    [void]WaitAny(){
        $run = @(Get-Job).id
        $myevent = ''
        While(!$myevent){
            #
            $myevent = Wait-Event -timeout 1
            if ($myevent){
                break
            }
            #
            if ($run){
                $myevent = Wait-Job -id $run -Any -Timeout 1
            }
            #
         }
         #
         if (($myevent[0].GetType()).Name -match 'job'){
            $this.CheckCompletedWorkers()
         } else {
             $this.CheckCompletedEvents()
         }
         #
    }
    #
    [void]WaitAny($run){
        #
        $myevent = ''
        While(!$myevent){
            #
            $myevent = Wait-Event -timeout 1
            if ($myevent){
                break
            }
            $myevent = Wait-Job -id $run -Any -Timeout 1
            #
        }
        #
    }
    #
    [void]WaitTask(){
        #
        $run = @(Get-Job | Where-Object { $_.State -eq 'Running'}).id
        if (!$this.workers -and $run){
            Wait-Job -id $run -Any
        }
        $this.CheckCompletedWorkers()
        #
    }
    #
    [void]CheckCompletedWorkers(){
        #
        $donejobs = Get-Job | 
            Where-Object { $_.State -eq 'Completed'}
        #
        if ($donejobs){
            $donejobs | ForEach-Object {
                #
                $this.checkpsexeclog($_.Name)
                $this.checkworkerlog($_)
                $this.updatecurrentworker($_.Name, 'IDLE')
                #
            }
            #
            $donejobs | Remove-Job
            #
        }
        #
    }
    #
    [void]CheckCompletedEvents(){
        #
        $events = get-event
        #
        while($events){
            #
            $currentevent = $events[0]
            remove-event -SourceIdentifier $currentevent.SourceIdentifier
            $this.handleAPevent($currentevent.SourceIdentifier)
            $events = get-event
            #
        }
        #
    }
    #
    [void]checkpsexeclog($jobname){
        #
        $psexeclog = $this.workertasklog($jobname)
        $psexectask = $this.getcontent($psexeclog)
        #
        if ($psexectask -match 'PsExec could not start powershell'){
            $this.writeoutput(" task could not be started on the remote machine, check psexec input")
            $this.writeoutput(" $psexectask")
            $this.CheckTaskLog($jobname, 'ERROR')
        } elseif ($psexectask -match 'error code 1'){
            $this.writeoutput(" powershell task did not exit successfully, might be a syntax error")
            $this.writeoutput(" $psexectask")
            $this.CheckTaskLog($jobname, 'ERROR')
        }
        #
        $this.removefile($psexeclog)
        #
    }
    #
    [void]checkworkerlog($job){
        #
        $taskid = ($job.Name, $job.PSBeginTime, $job.PSEndTime) -join '-'
        $output = $this.getcontent($this.workerlogfile($job.Name)) 
        #
        # if there is output and the last line does not match the Regex write lines 
        # to console 
        #
        if ($output -and $output[$output.count-1] -notmatch [regex]::escape($_.Name)) {
            $mymatches = $output -match [regex]::escape($_.Name)
            $idx = [array]::IndexOf($output, $mymatches[-1])
            $newerror = $output[($idx+1)..($output.count-1)]
            $this.writeoutput(" $taskid")
            $this.writeoutput(" $newerror")
        }
        #
        $this.CheckTaskLog($job.Name, 'WARNING')
        $this.removefile($this.workertaskfile($job.Name))
        [string]$logline = @("FINISH: ", $taskid,"`r`n") -join ''
        $this.popfile(($this.workerlogfile($job.Name)), $logline)
        #
    }
    #
    [void]updatecurrentworker($jobname, $newstatus){
        #
        $currentworkerstrings = $this.parseJobName($jobname)
        $currentworker = $this.worker_data | Where-Object {
            $_.server -contains $currentworkerstrings[0] -and 
            $_.location -contains $currentworkerstrings[1] -and 
            $_.module -contains $currentworkerstrings[2]
        }
        $currentworker.status = $newstatus
        #
    }
    <# -----------------------------------------
     CheckTaskLog
     Check the task specific logs for start 
     messages with no stops
     ------------------------------------------
     Usage: $this.CheckTaskLog()
    ----------------------------------------- #>
    [void]CheckTaskLog($jobname, $level){
        #
        # open the workertaskfile
        #
        $mtask = $this.getcontent($this.workertaskfile($jobname))
        #
        # parse out necessary information
        #
        $file1 = $mtask -split 'module'
        $ID = ($file1[4] -split '} 2')[0]
        $ID = ($ID -split '-') -replace '"', ''
        $cmodule = $ID[0].trim()
        #
        $cslideid = ($ID -match 'slideid')
        if ($cslideid){
            $cslideid = ($cslideid -replace 'slideid', '').trim()
        }
        #
        $cproject = ($ID -match 'project')
        if ($cproject){
            $cproject = ($cproject -replace 'project', '').trim()
        }
        #
        $cbatchid = ($ID -match 'batchid')
        if ($cbatchid){
            $cbatchid = ($cbatchid -replace 'batchid', '').trim()
        }
        #
        # create a logging object and check the
        # log for a finishing message
        #
        try{
            if ($cmodule -match 'batch'){
                $log = logger -mpath:$this.mpath $cmodule -batchid:$cbatchid -project:$cproject
            } else {
                $log = logger -mpath:$this.mpath -module:$cmodule -slide:$cslideid
            }
        } catch {
            $this.writeoutput(" "+$_.Exception.Message)
            $this.writeoutput(" $ID")
            $this.writeoutput(" SlideID: $cslideid")
            $this.writeoutput(" BatchID: $cbatchid")
            $this.writeoutput(" ProjectID: $cproject")
            return
        }
        #
        if(!($this.checklog($log, $false))){
            if ($level -match 'ERROR'){
                 $log.error('Task did not seem to complete correctly, check results')
            } elseif ($level -match 'WARNING'){
                 $log.warning('Task did not seem to complete correctly, check results')
            }
            $log.finish($cmodule)
        }
    }
    <# -----------------------------------------
     Getlogin
     puts credentials in a string format for psexec 
     ------------------------------------------
     Usage: $this.Getlogin()
    ----------------------------------------- #>
    [array]Getlogin(){
        #
        $username = $this.login.UserName
        $password = $this.login.GetNetworkCredential().Password
        return @($username, $password)
        #
    }
     <# -----------------------------------------
     DefJobName
     define the job name as server-location-module
     ------------------------------------------
     Usage: $this.DefJobName($currentworker)
    ----------------------------------------- #>
    [string]DefJobName($currentworker){
        return ($currentworker.server,
            $currentworker.location, $currentworker.module) -join ';'     
    }
    <# -----------------------------------------
     parseJobName
     define the job name as server-location-module
     ------------------------------------------
     Usage: $this.parseJobName($currentworker)
    ----------------------------------------- #>
    [array]parseJobName($jobname){
        return ($jobname -split ';')   
    }
    <# -----------------------------------------
     workertasklog
     the log for the psexc job. errors in this
     file indicate errors in launching the 
     psexc and not the module itself
    ----------------------------------------- #>
    [string]workertasklog($jobname){
        return (
            $this.workerloglocation, $jobname, '-taskfile-job.log' -join ''
        )
    }
     <# -----------------------------------------
     workertaskfile
     the task to launch. the string task is 
     created in $this.workertaskfile
    ----------------------------------------- #>
    [string]workertaskfile($jobname){
        return (
            $this.workerloglocation, $jobname, '-taskfile.ps1' -join ''
        )
    }
     <# -----------------------------------------
     workerlogfile
     the log file for the powershell module
     errors in this file indicate unhandled
     exceptions in the powershell module or 
     launching code. 
    ----------------------------------------- #>
    [string]workerlogfile($jobname){
        return ($this.workerloglocation, $jobname, '.log' -join '')
    }
    <# -----------------------------------------
     DefCurrentWorkerip
     define the current worker IP address. The
     server names can be used as aliases but the 
     jhu bki VMs were added to the domain as
     vminform## rather than their unique names
     VM_inForm## that the were created
    ----------------------------------------- #> #
    [string]DefCurrentWorkerip($currentworker){
        #
        if ($currentworker.location -match 'VM'){
            $currentworkerip = ($currentworker.location -replace '_', '').tolower()
        } else {
            $currentworkerip = $currentworker.server
        }
        return $currentworkerip
        #
    }
}