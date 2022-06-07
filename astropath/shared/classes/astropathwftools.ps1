#
class astropathwftools : sampledb {
    #
    [Pscredential]$login
    [string]$workerloglocation = '\\' + $env:ComputerName +
        '\c$\users\public\astropath\' 
    [array]$apworkerheaders = @('server','location','module','message','date')
    [array]$apworkerlog
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
        $idleworkers = $this.worker_data |
            where-object {$_.status -match 'IDLE'}
        foreach ($worker in @(0..($idleworkers.Length - 1))){
            #
            $jobname = $this.defjobname($idleworkers[$worker])
            $workertasklog = $this.workertasklog($jobname)
            $workertaskfile = $this.workertaskfile($jobname)
            $processid = $this.checkprocessid($jobname)
            #
            if ($processid -ne 0){
                $this.writeoutput("     Orphaned job found: $workertasklog")
                $this.StartOrphanMonitor($jobname, $processid)
                $idleworkers[$worker].Status = 'RUNNING'
                $this.writeoutput(
                        "     Orphaned job not completed: $workertasklog ... created processid watchdog"
                    )
                
            }  elseif (test-path $workertasklog) {
                #
                $this.writeoutput("     Orphaned job found: $workertasklog")
                $this.removefile($workertasklog)
                $this.CheckTaskLog($jobname, 'ERROR')
                $this.removefile($workertaskfile)
                $idleworkers[$worker].Status = 'IDLE'
                $this.writeoutput(
                    "     Orphaned job completed and cleared: $workertasklog"
                )
                #
            }               
          
        }
        #
    }
    #
    [int]checkprocessid($jobname){
        #
        $this.importaplog()
        $logline = $this.selectaploglines($jobname).message -match 'processid'
        if (!$logline){
            return 0
        }
        $processid = ($logline -split 'processid: ')[1]
        #
        $jobid = $this.parsejobname($jobname)
        $currentworker = $this.jobtoworker($jobid)
        $ip = $this.defcurrentworkerip($currentworker)
        #
        if (
            $env:computername -match $currentworker.server -and 
            $currentworker.module -notmatch 'vminform'
        ){
            $proc = get-process -id $processid -ErrorAction silentlycontinue
        } else {
            try {
                $proc = invoke-command -computername $ip -credential $this.login -EA Stop `
                    -scriptblock {get-process -id $using:processid -ErrorAction silentlycontinue}
            } catch {
                $this.enableremoting($ip)
                $proc = invoke-command -computername $ip -credential $this.login -EA Stop `
                    -scriptblock {get-process -id $using:processid -ErrorAction silentlycontinue} 
            }
        }
        #
        if ($proc){
            return $processid
        }
        #
        return 0
        #
    }
    #
    [void]enableremoting($cname){
        #
        $SessionArgs = @{
            ComputerName  = $cname
            Credential    = $this.login
            SessionOption = New-CimSessionOption -Protocol Dcom
        }
        $MethodArgs = @{
            ClassName     = 'Win32_Process'
            MethodName    = 'Create'
            CimSession    = New-CimSession @SessionArgs
            Arguments     = @{
                CommandLine = "powershell Start-Process powershell -ArgumentList 'Enable-PSRemoting -Force'"
            }
        }
        Invoke-CimMethod @MethodArgs
        #
    }
    #
    <# -----------------------------------------
     StartOrphanMonitor
     Start an orphaned job monitor 
     ------------------------------------------
     Usage: $this.StartOrphanMonitor()
    ----------------------------------------- #>
    [void]StartOrphanMonitor($jobname, $processid){
        #
        # start a new job monitoring this file with the same job name 
        #
        $jobid = $this.parsejobname($jobname)
        $currentworker = $this.jobtoworker($jobid)
        $ip = $this.defcurrentworkerip($currentworker)
        #
        $sb = {
            param($processid)
            wait-process -id $processid -ea silentlycontinue
        }
        #
        if (
            $env:computername -match $currentworker.server -and 
            $currentworker.module -notmatch 'vminform'
        ){
            #
            $myparameters = @{
                ScriptBlock = $sb
                ArgumentList = $processid
                name = $jobname
            }
            Start-Job @myparameters
            #
         } else {
            #
            invoke-command -computername $ip -credential $this.login `
                -scriptblock $sb -asjob -jobname $jobname -ArgumentList $processid
            #
        }    
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
                $currentworker, $currentworkers = $currentworkers
                if (!($this.fastping($currentworker))){
                    $this.writeoutput(('WARNING:', 
                        $currentworker.server, $currentworker.location,
                        'is set to ON but state is OFF!' -join ' '))
                    continue
                }
                #
                $currenttask = $cqueue.dequeue()
                #
                $this.writeoutput(("     Launching Task on:", 
                    $currentworker.server, $currentworker.location -join ' '))
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
        $this.writeoutput("     $currenttask")
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
        $row.StartDate = $this.getformatdate()
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
        $currenttasktowrite = (' Import-Module "', $this.coderoot(), '" -ea stop
        $output = & {LaunchModule -mpath:"', $this.mpath, $currenttaskinput,
            '" -tasklogfile "', $this.workerlogfile(),'" -jobname "', $jobname,'"} 2>&1
        UpdateProcessingLog -logfile "', $this.workerlogfile(),'" -jobname "', $jobname,
            '" -sample $output -erroroutput $output.output') -join ''
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
        #
        $this.popfile($this.workerlogfile(),
            $this.aplogstart($jobname, $jobname))
        #
        $this.popfile($this.workerlogfile(),
             $this.aploginfo($jobname, ($currenttask -join ' ')))
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
            $this.clearevents($currentevent)
            $this.handleAPevent($currentevent.SourceIdentifier)
            $events = get-event
            #
        }
        #
        $this.CheckCompletedWorkers()
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
        $this.importaplog()
        $this.writeoutput($job.Name)
        $this.selectaploglines($job.Name) | 
            ForEach-Object {
                $this.writeoutput($_.Message)
            }
        #
        $this.CheckTaskLog($job.Name, 'WARNING')
        $this.removefile($this.workertaskfile($job.Name))
        #
        $this.popfile($this.workerlogfile(),
          $this.aplogfinish($job.name, $job.name))
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
            $currentworker.location, $currentworker.module) -join '.'     
    }
    <# -----------------------------------------
     parseJobName
     define the job name as server-location-module
     ------------------------------------------
     Usage: $this.parseJobName($currentworker)
    ----------------------------------------- #>
    [array]parseJobName($jobname){
        return ($jobname -split '\.')   
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
    [string]workerlogfile(){
        return ($this.workerloglocation, 'astropath-workers.log' -join '')
    }
    #
    [string]aplogerror($jobname, $message){
        return $this.formatwlogfile($jobname, ('ERROR:', $message -join ' '))
    }
    #
    [string]aplogwarning($jobname, $message){
        return $this.formatwlogfile($jobname, ('WARNING:', $message -join ' '))
    }
    #
    [string]aploginfo($jobname, $message){
        return $this.formatwlogfile($jobname, ('INFO:', $message -join ' '))
    }
    #
    [string]aplogstart($jobname, $message){
        return $this.formatwlogfile($jobname, ('START:', $message -join ' '))
    }
    #
    [string]aplogfinish($jobname, $message){
        return $this.formatwlogfile($jobname, ('FINISH:', $message -join ' '))
    }
    #
    [string]formatwlogfile($jobname, $message){
        return (
            (($this.parsejobname($jobname) -join ';'),
             $message, $this.getformatdate() -join ';') + "`r`n"
        )
    }
    #
    [void]importaplog(){
        $this.apworkerlog = $this.opencsvfile(
            $this.workerlogfile(), ';', $this.apworkerheaders
        )
    }
    #
    [array]selectaploglines($jobname){
        #
        $jobname1 = $this.parseJobName($jobname)
        return (
            $this.apworkerlog |
                Where-Object {
                    $_.server -contains $jobname1[0] -and 
                    $_.location -contains $jobname1[1] -and 
                    $_.module -contains $jobname1[2] -and 
                    $_.Date -ge ($this.selectaploglines($jobname, 'START').date)
                }
        )
        #
    }
    #
    [array]selectaploglines($jobname, $status){
        #
        $jobname = $this.parseJobName($jobname)
        return (
            $this.apworkerlog |
                Where-Object {
                    $_.server -contains $jobname[0] -and 
                    $_.location -contains $jobname[1] -and 
                    $_.module -contains $jobname[2] -AND
                    $_.Message -match ('^' + $status)
                } |
                Select-Object -Last 1
        )
        #
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
    #
    [PSCustomObject]jobtoworker($jobname){
        #
        $hash = @{ 
            server =$jobname[0];
            location = $jobname[1];
            module = $jobname[2]
        }
        #
        return [PSCustomObject]$hash
        #
    }
    #
    [switch]fastping($currentworker){
        #
        $obj = New-Object System.Net.NetworkInformation.Ping
        return (
            (
                $obj.send($this.defcurrentworkerip($currentworker), 2000)
            ).status -match 'Success'
        )
        #
    }
}