##
# dispatcher tools
#
class DispatcherTools : queue {
    #
    [PSCredential]$cred
    [string]$workerloglocation = '\\' + $env:ComputerName +
        '\c$\users\public\astropath\'    
    #
    DispatcherTools($mpath, $module, $cred) : base($mpath, $module){
        #
        $this.init($cred)
        $this.Run()
        #
    }
    #
    DispatcherTools($mpath, $module, $project, $cred) : base($mpath, $module, $project){
        #
        $this.init($cred)
        $this.Run()
        #
    }
    #
    Dispatchertools($mpath, $module, $project, $slideid, $cred) : base($mpath, $module, $project, $slideid){
        #
        $this.init($cred)
        $this.Run()
        #
    }
    #
    Dispatchertools($mpath, $module, $project, $slideid, $test, $cred) : base($mpath, $module){
        # constructor for testing code will not run
        $this.init($cred)
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
                #
                try { 
                    $fileInfo = New-Object System.IO.FileInfo $workertasklog
                    $fileStream = $fileInfo.Open([System.IO.FileMode]::Open)
                    $fileStream.Dispose()
                    break
                } catch {
                    #
                    $myevent = wait-event $SI  
                    remove-event $myevent.EventIdentifier
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
        $task = $this.getcontent($this.workertaskfile($jobname))
        #
        # parse out necessary information
        $file1 = $task -split 'module'
        $ID = ($file1[4] -split '} 2')[0]
        $ID = ($ID -split '-') -replace '"', ''
        $cmodule = $ID[0].trim()
        #
        $cslideid = ($ID -match 'slideid')
        $cslideid = ($cslideid -replace 'slideid', '').trim()
        #
        $cproject = ($ID -match 'project')
        $cproject = ($cproject -replace 'project', '').trim()
        #
        $cbatchid = ($ID -match 'batchid')
        $cbatchid = ($cbatchid -replace 'batchid', '').trim()
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
            Write-Host $_.Exception.Message
            Write-Host $ID
            Write-Host 'SlideID:' $cslideid
            Write-Host 'BatchID:' $cbatchid
            Write-Host 'Project:' $cproject
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
     GetCreds
     puts credentials in a string format for psexec 
     ------------------------------------------
     Usage: $this.GetCreds()
    ----------------------------------------- #>
    [array]GetCreds(){
        #
        $username = $this.cred.UserName
        $password = $this.cred.GetNetworkCredential().Password
        return @($username, $password)
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
                    $output.popfile("',$this.workerlogfile($jobname),'", ("ERROR: " + $count + "`r`n"))
                    $output.popfile("',$this.workerlogfile($jobname),'", ("  " + $_.Exception.Message  + "`r`n"))
                    $s = $_.ScriptStackTrace.replace("at", "`t at")
                    $output.popfile("',$this.workerlogfile($jobname),'", ($s + "`r`n"))
                    $count += 1
                }
                #
            } else {
                $output.popfile("',$this.workerlogfile($jobname),'", "Completed Successfully `r`n")
            }') -join ''
        #
        $this.SetFile($this.workertaskfile($jobname), $currenttasktowrite)
        #
    }
    #
    [void]PrepareWorkerFiles($currenttask, $jobname, $currentworker){
        #
        $this.createdirs($this.workerloglocation)
        $currenttaskinput = $this.addedargs($currenttask, $currentworker)
        $this.buildtaskfile($jobname, $currenttaskinput)
        #
    }
    #
    [string]addedargs($currenttask, $currentworker){
        #
        if ($this.module -match 'vminform'){
            $currenttaskinput = '" -module ', $this.module,
                ' -project ', $currenttask[0], ' -slideid ', $currenttask[1],
                ' -antibody ', $currenttask[2], ' -algorithm ', $currenttask[3],
                ' -informvers ', $this.informvers -join '"'
        } elseif ($this.module -match 'batch') {
            $currentworkerstring = '\\' + $currentworker.server + '\' + $currentworker.location
            $currenttaskinput = '" -module ', $this.module,
                ' -project ', $currenttask[0], ' -batchid ', $currenttask[1],
                ' -processloc ', $currentworkerstring -join '"'
        } else {
            $currentworkerstring = '\\' + $currentworker.server + '\' + $currentworker.location
            $currenttaskinput = '" -module ', $this.module, 
                ' -project ', $currenttask[0], ' -slideid ', $currenttask[1],
                ' -processloc ', $currentworkerstring -join '"'
        }
        #
        return $currenttaskinput
        #
    }
    #
    [string]workerlogfile($jobname){
        return $this.workerloglocation+$jobname+'.log'
    }
    #
    [string]workertaskfile($jobname){
        return $this.workerloglocation+$jobname+'-taskfile.ps1'
    }
    #
    [string]workertasklog($jobname){
        return $this.workerloglocation+$jobname+'-taskfile-job.log'
    }
    #
    [void]checkpsexeclog($jobname){
        #
        $psexeclog = $this.workertasklog($jobname)
        $workertaskfile = $this.workertaskfile($jobname)
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
            $matches = $output -match [regex]::escape($_.Name)
            $idx = [array]::IndexOf($output, $matches[-1])
            $newerror = $output[($idx+1)..($output.count-1)]
            write-host $taskid
            Write-host $newerror
        }
        #
        $this.CheckTaskLog($job.Name, 'WARNING')
        $this.removefile($this.workertaskfile($job.Name))
        [string]$logline = @("FINISH: ", $taskid,"`r`n") -join ''
        $this.popfile(($this.workerlogfile($job.Name)), $logline)
        #
    }
    #
}