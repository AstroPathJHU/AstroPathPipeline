##
# launch a queue for a provide module with a provided mpath and credentials
#
class Dispatcher : DispatcherTools {
    #
    [array]$running
    [array]$workers
    #
    Dispatcher($mpath, $module, $cred) : base($mpath, $module, $cred){
        #
        $this.init($cred)
        $this.Run()
        #
    }
    #
    Dispatcher($mpath, $module, $project, $cred) : base($mpath, $module, $project, $cred){
        #
        $this.init($cred)
        $this.Run()
        #
    }
    #
    Dispatcher($mpath, $module, $project, $slideid, $cred) : base($mpath, $module, $project, $slideid, $cred){
        #
        $this.init($cred)
        $this.Run()
        #
    }
    #
    Dispatcher($mpath, $module, $project, $slideid, $test, $cred) : base($mpath, $module, $project, $slideid, $test, $cred){
        #
        $this.init($cred)
        #
    }
    <# -----------------------------------------
     Run
     runs the bulk of the code
     1. initializes the worker list
     2. get running jobs
     3. check for new tasks
     4. distribute tasks
     5. wait for tasks to finish
     6. check for new tasks
     7. repeat
     ------------------------------------------
     Usage: $this.Run()
    ----------------------------------------- #>
    [void]Run(){
        #
        while(1){
            Write-Host "." -ForegroundColor Yellow
            $this.InitializeWorkerlist()
            $this.GetRunningJobs()
            Write-Host " Checking for tasks" -ForegroundColor Yellow
            $this.checknew()
            $this.DistributeTasks()
            $this.WaitTask()
            $this.checknew()
        }
        #
    }
    <# -----------------------------------------
     InitializeWorkerlist
     Builds the worker list for the desired 
     module from the AstroPathHPFsWlocs file
     ------------------------------------------
     Usage: $this.InitializeWorkerlist()
    ----------------------------------------- #>
    [void]InitializeWorkerlist(){
        #
        # get list of workers machines that are on
        #
        $this.workers = $this.OpenCSVFile($this.mpath+'\AstroPathHPFWLocs.csv') |
                             Where-Object {$_.module -eq $this.module}
        
        #
        Write-Host " Starting-Task-Distribution" -ForegroundColor Yellow
        write-host " Current Computers for Processing:" -ForegroundColor Yellow
        write-host " " ($this.workers | 
            Format-Table  @{Name="module";Expression = { $_.module }; Alignment="center" },
                            @{Name="server";Expression = { $_.server }; Alignment="center" },
                            @{Name="location";Expression = { $_.location }; Alignment="center" } |
            Out-String).Trim() -ForegroundColor Yellow
        Write-Host "  ." -ForegroundColor Yellow
        #
    }
    <# -----------------------------------------
     GetRunningJobs
     Removes the currently running jobs from the
     worker list using the unique job names
     ------------------------------------------
     Usage: $this.GetRunningJobs()
    ----------------------------------------- #>
    [void]GetRunningJobs(){
        #
        $this.running = @(Get-Job | 
            Where-Object { $_.State -eq 'Running' -and $_.Name -match $this.module})
        if ($this.running){
            $this.running.Name | FOREACH-Object {
               $CC = $_
               $this.workers = $this.workers | 
                where-object {(($_.server, $_.location, $_.module) -join('-')) -ne  $CC}
            }
        }
        #
        $this.checkfororphanedtasks()
        #
    }
    <# -----------------------------------------
     CheckForOrphanedTasks
     Check for orphaned tasks from a previously
     running instance of astropath. If it finds
     an orphaned task it will create a new job
     with the orphaned jobs name that peridoucally
     checks for access. when access is confirmed
     the job exists and allows the worker to 
     seemlessly jump back into the workflow.
     ------------------------------------------
     Usage: $this.CheckForOrphanedTasks()
    ----------------------------------------- #>
    [void]CheckForOrphanedTasks(){
        #
        $workerstouse = $this.workers
        #
        $this.workers | foreach-object{
            #
            $currentjob = $_
            $jobname = $this.defjobname($currentjob)
            $workertasklog = $this.workertasklog($jobname)
            $workertaskfile = $this.workertaskfile($jobname)
            #
            if (test-path $workertasklog){
                $fileInfo = New-Object System.IO.FileInfo $workertasklog
                try {
                    #
                    # check if I can access and remove the worker task file
                    #
                    $fileStream = $fileInfo.Open([System.IO.FileMode]::Open)
                    $fileStream.Dispose()
                    $this.removefile($workertasklog)
                    $this.CheckTaskLog($jobname, 'ERROR')
                    $this.removefile($workertaskfile)
                    #
                }catch {
                    #
                    $this.StartOrphanMonitor($jobname)
                    #
                    # remove the worker from the list 
                    #
                    $workerstouse = $workerstouse | 
                        Where-Object {($_.server -ne $currentjob.server `
                            -or $_.location -ne $currentjob.location)}
                }
           }
        }
        #
        $this.workers = $workerstouse
        #
    }
   <# -----------------------------------------
     CheckNew
     Checks for new tasks and rebuilds the worker
     list
     ------------------------------------------
     Usage: $this.CheckNew()
    ----------------------------------------- #>
    [void]CheckNew(){
        #
        while (1){
            $this.ExtractQueue()
            if (!($this.cleanedtasks)){
                Write-Host " No new samples to process." -ForegroundColor Yellow
                Write-Host " Sleeping for 10 minutes." -ForegroundColor Yellow
                Start-Sleep -s (10 * 60)
                Write-Host " Checking for tasks" -ForegroundColor Yellow
            } else {
                Write-Host " Tasks Found" -ForegroundColor Yellow
                break
            }
        }
        #
        $this.CheckCompletedWorkers()
        #
    }
    <# -----------------------------------------
     DistributeTasks
     distribute tasks to the workers 
     ------------------------------------------
     Usage: $this.DistributeTasks()
    ----------------------------------------- #>
    [void]DistributeTasks(){
        #
        While($this.workers -and $this.cleanedtasks){
            #
            $currentworker, $this.workers = $this.workers 
            $tasktomatch, $this.originaltasks = $this.originaltasks
            $currenttask, $this.cleanedtasks = $this.cleanedtasks
            #
            Write-Host "  Launching Task on:" $currentworker.server $currentworker.location `
                    -ForegroundColor Yellow
            Write-Host "    "$currenttask -ForegroundColor Yellow
            #
            $this.launchtask($currenttask, $currentworker)
            $this.UpdateQueue($currenttask, $currentworker, $tasktomatch)
            #
        }
        #
    }
    <# -----------------------------------------
     LaunchTask
     Launch a task on a current worker
     ------------------------------------------
     Usage: $this.LaunchTask()
    ----------------------------------------- #>
    [void]LaunchTask($currenttask, $currentworker){
        #
        $creds = $this.GetCreds()       
        $currentworkerip = $this.defcurrentworkerip($currentworker)
        $jobname = $this.defjobname($currentworker)
        $this.PrepareWorkerFiles($currenttask, $jobname, $currentworker)
        #
        if ($currentworker.location -match 'VM'){
            $myscriptblock = {
                param($username, $password, $currentworkerip, $workertaskfile)
                psexec -i -nobanner -accepteula -u $username -p $password \\$currentworkerip `
                    pwsh -noprofile -executionpolicy bypass -command "$workertaskfile" `
                    *>> ($workertaskfile -replace '.ps1', '-job.log')
            }
        } else {
            $myscriptblock = {
                param($username, $password, $currentworkerip, $workertaskfile)
                psexec -nobanner -accepteula -u $username -p $password \\$currentworkerip `
                    pwsh -noprofile -WindowStyle Hidden -executionpolicy bypass -command "$workertaskfile" `
                    *>> ($workertaskfile -replace '.ps1', '-job.log')
            }
        }
        #
        $myparameters = @{
            ScriptBlock = $myscriptblock
            ArgumentList = $creds[0], $creds[1], $currentworkerip, $this.workertaskfile($jobname)
            name = $jobname
            }
        #
        Start-Job @myparameters
        $taskid = ($jobname, (Get-Date)) -join '-'
        [string]$logline = @("START: ", $taskid,"`r`n") -join ''
        $this.popfile($this.workerlogfile($jobname), $logline)
        #
    }
    #
    [void]WaitAny(){
        $run = @(Get-Job | Where-Object { $_.State -eq 'Running'}).id
        $myevent = ''
        if (!$this.workers -and $run){
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
            if (($myevent[0].GetType()).Name -match 'job'){
                $this.CheckCompletedWorkers()
            } else {
                $this.CheckCompletedEvents()
            }
            #
        } else {
            $this.CheckCompletedEvents()
        }
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
            $donejobs | Remove-Job
            $donejobs | ForEach-Object {
                #
                $this.checkpsexeclog($_.Name)
                $this.checkworkerlog($_)
                #
            }
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
            $this.handleAPevent($currentevent)
            $events = get-event
            #
        }
        #
    }
    #
}