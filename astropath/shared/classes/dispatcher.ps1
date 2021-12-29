##
# launch a queue for a provide module with a provided mpath and credentials
#
class Dispatcher : queue{
    #
    [switch]$new
    [array]$running
    [array]$workers
    [PSCredential]$cred
    [string]$workerloglocation = '\\' + $env:ComputerName +
        '\c$\users\public\astropath\'
    #
    Dispatcher($mpath, $module, $cred):base($mpath, $module){
        #
        $this.init($cred)
        $this.Run()
        #
    }
    #
    Dispatcher($mpath, $module, $project, $cred):base($mpath, $module, $project){
        #
        $this.init($cred)
        $this.Run()
        #
    }
    #
    Dispatcher($mpath, $module, $project, $slideid, $cred):base($mpath, $module, $project, $slideid){
        #
        $this.init($cred)
        $this.Run()
        #
    }
    <# -----------------------------------------
     Run
     actuall runs the bulk of the code
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
    #
    # initialize the code
    #
    [void]init($cred){
        $this.cred = $cred
        Write-Host "Starting the AstroPath Pipeline" -ForegroundColor Yellow
        Write-Host ("Module: " + $this.module) -ForegroundColor Yellow
        Write-Host ("Username: " + $this.cred.UserName) -ForegroundColor Yellow
    }
    <# -----------------------------------------
     initepy
     checks that conda is installed and a functional
     command
     ------------------------------------------
     Usage: $this.initepy()
    ----------------------------------------- #>
    [void]initepy(){
        Write-Host "Initializing\updating the conda environment" -ForegroundColor Yellow
        $this.checkconda()
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
            $this.running.Name | FOREACH {
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
            $workertasklog = $this.workerloglocation+$jobname+'-taskfile-job.log'
            $workertaskfile = $this.workerloglocation+$jobname+'-taskfile.ps1'
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
                    $this.CheckTaskLog($workertaskfile, 'ERROR')
                    $this.removefile($workertaskfile)
                }catch {
                    #
                    # start a new job monitoring this file with the same job name 
                    #
                    $myscriptblock = {
                        param($workertaskfile)
                        while (1) {
                            try { 
                                $fileInfo = New-Object System.IO.FileInfo $workertaskfile
                                $fileStream = $fileInfo.Open([System.IO.FileMode]::Open)
                                $fileStream.Dispose()
                                break
                            } catch {
                                Start-Sleep -s (10 * 60)
                            }
                        }
                    }
                    #
                    $myparameters = @{
                        ScriptBlock = $myscriptblock
                        ArgumentList = $workertaskfile
                        name = $jobname
                    }
                    Start-Job @myparameters
                    #
                    # remove the worker from the list 
                    #
                    $workerstouse = $workerstouse | 
                        Where-Object {($_.server -ne $currentjob.server -or $_.location -ne $currentjob.location)}
                }
           }
        }
        #
        $this.workers = $workerstouse
        #
    }
    <# -----------------------------------------
     CheckTaskLog
     Check the task specific logs for start 
     messages with no stops
     ------------------------------------------
     Usage: $this.CheckTaskLog()
    ----------------------------------------- #>
    [void]CheckTaskLog($workertaskfile, $level){
        #
        # open the workertaskfile
        #
        $task = $this.getcontent($workertaskfile)
        #
        # parse out necessary information
        #
        $ID = ($task -split '-stringin:')[2]
        $ID = ($ID -split '} 2')[0]
        $ID = $ID -split '-'
        #
        # create a logging object and check the
        # log for a finishing message
        #
        $log = [mylogger]::new($this.mpath, $this.module, $ID[1])
        #
        if ($this.module -match 'batch'){
            $log.slidelog = $log.mainlog
        }
        #
        if(!($this.checklog($log, $false))){
            if ($level -match 'ERROR'){
                 $log.error('Task did not seem to complete correctly, check results')
            } elseif ($level -match 'WARNING'){
                 $log.warning('Task did not seem to complete correctly, check results')
            }
            $log.finish($this.module)
        }
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
            Write-Host "  Launching Task on:" $currentworker.server $currentworker.location -ForegroundColor Yellow
            Write-Host "    "$currenttask -ForegroundColor Yellow
            #
            $this.launchtask($currenttask, $currentworker)
            $this.UpdateQueue($currenttask, $currentworker, $tasktomatch)
            #
        }
        #
    }
    #
    [array]GetCreds(){
        $username = $this.cred.UserName
        $password = $this.cred.GetNetworkCredential().Password
        return @($username, $password)
    }
    #
    [void]LaunchTask($currenttask, $currentworker){
        #
        $creds = $this.GetCreds()       
        $currentworkerip = $this.defcurrentworkerip($currentworker)
        $jobname = $this.defjobname($currentworker)
        $workertaskfile = $this.PrepareWorkerFiles($currenttask, $jobname, $currentworker)
        #
        if ($currentworker.location -match 'VM'){
            $myscriptblock = {
                param($username, $password, $currentworkerip, $workertaskfile)
                psexec -i -nobanner -accepteula -u $username -p $password \\$currentworkerip `
                    powershell -noprofile -executionpolicy bypass -command "$workertaskfile" `
                    *>> ($workertaskfile -replace '.ps1', '-job.log')
            }
        } else {
            $myscriptblock = {
                param($username, $password, $currentworkerip, $workertaskfile)
                psexec -nobanner -accepteula -u $username -p $password \\$currentworkerip `
                    powershell -noprofile -WindowStyle Hidden -executionpolicy bypass -command "$workertaskfile" `
                    *>> ($workertaskfile -replace '.ps1', '-job.log')
            }
        }
        #
        $myparameters = @{
            ScriptBlock = $myscriptblock
            ArgumentList = $creds[0], $creds[1], $currentworkerip, $workertaskfile
            name = $jobname
            }
        Start-Job @myparameters
        #
        $taskid = ($jobname, (Get-Date)) -join '-'
        [string]$logline = @("START: ", $taskid,"`r`n") -join ''
        $this.popfile(($this.workerloglocation+$jobname+'.log'), $logline)
        #
    }
    #
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
    [string]DefJobName($currentworker){
         $jobname = ($currentworker.server, $currentworker.location, $this.module) -join '-'     
         return $jobname
    }
    #
    [string]PrepareWorkerFiles($currenttask, $jobname, $currentworker){
        #
        if (!(test-path $this.workerloglocation)){
            new-item $this.workerloglocation -itemtype "directory" -EA STOP | Out-NULL
        }
        #
        $workertaskfile = $this.workerloglocation+$jobname+'-taskfile.ps1'
        $workerlogfile = $this.workerloglocation+$jobname+'.log'
        #
        if ($this.module -match 'vminform'){
            $currenttaskinput = ($currenttask, $this.informvers) -join ',' -replace ',','-'
        } else {
            $currentworkerstring = '\\' + $currentworker.server + '\' + $currentworker.location
            $currenttaskinput = ($currenttask, $currentworkerstring) -join ',' -replace ',','-'
        }
        #
        $this.buildtaskfile($workertaskfile, $currenttaskinput, $workerlogfile)
        return $workertaskfile
        #
    }
    #
    [void]buildtaskfile($workertaskfile, $currenttaskinput, $workerlogfile){
        #
        $currenttasktowrite = (' Import-Module "', $this.coderoot(), '"
                        $output = & {LaunchModule -mpath:"', `
                        $this.mpath, '" -module:"', $this.module, '" -stringin:"', `
                        $currenttaskinput,'"} 2>&1
                        $tool = [sharedtools]::new()
                    if ($output -ne 0){ 
                        #
                        $count = 1
                        #
                        $output | Foreach-object {
                            $tool.popfile("',$workerlogfile,'", ("ERROR: " + $count))
                            $tool.popfile("',$workerlogfile,'", ("  " + $_.Exception.Message))
                            $s = $_.ScriptStackTrace.replace("at", "`t at")
                            $tool.popfile("',$workerlogfile,'", $s)
                            $count += 1
                        }
                        #
                    } else {
                        $tool.popfile("',$workerlogfile,'", "Completed Successfully")
                    }') -join ''
        #
        $this.SetFile($workertaskfile, $currenttasktowrite)
        #
    }
    #
    [void]WaitTask(){
        #
        $run = @(Get-Job | Where-Object { $_.State -eq 'Running'  -and $_.Name -match $this.module}).id
        if (!$this.workers -and $run){
            Wait-Job -id $run -Any
        }
        $this.CheckCompletedWorkers()
        #
    }
    #
    [void]CheckCompletedWorkers(){
        #
        $donejobs = Get-Job | Where-Object { $_.State -eq 'Completed'  -and $_.Name -match $this.module}
        if ($donejobs){
            $donejobs | Remove-Job
            $donejobs | ForEach {
                #
                # first check the psexec log
                #
                $psexeclog = $this.workerloglocation+$_.Name+'-taskfile-job.log'
                $workertaskfile = $this.workerloglocation+$_.Name+'-taskfile.ps1'
                $psexectask = $this.getcontent($psexeclog)
                #
                if ($psexectask -match 'PsExec could not start powershell'){
                    Write-Host 'task could not be started on the remote machine, check psexec input'
                    Write-Host $psexectask
                    $this.CheckTaskLog($workertaskfile, 'ERROR')
                } elseif ($psexectask -match 'error code 1'){
                    Write-Host 'powershell task did not exit successfully, might be a syntax error'
                    Write-Host $psexectask
                    $this.CheckTaskLog($workertaskfile, 'ERROR')
                }
                #
                $this.removefile($psexeclog)
                #
                # write new os errors to the console and add finishing line to os log
                #
                $taskid = ($_.Name, $_.PSBeginTime, $_.PSEndTime) -join '-'
                $output = $this.getcontent($this.workerloglocation+$_.Name+'.log') 
                #
                if ($output -and $output[$output.count-1] -notmatch [regex]::escape($_.Name)) {
                    $matches = $output -match [regex]::escape($_.Name)
                    $idx = [array]::IndexOf($output, $matches[-1])
                    $newerror = $output[($idx+1)..($output.count-1)]
                    write-host $taskid
                    Write-host $newerror
                }
                #
                $this.CheckTaskLog($workertaskfile, 'WARNING')
                $this.removefile($workertaskfile)
                [string]$logline = @("FINISH: ", $taskid,"`r`n") -join ''
                $this.popfile(($this.workerloglocation+$_.Name+'.log'), $logline)
                #
            }
        }
        #
    }
    #
}