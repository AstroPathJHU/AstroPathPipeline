##
# launch a queue for a provide module with a provided mpath and credentials

class Queue : sharedtools{
    #
    [Array]$tasks
    [Array]$tasks1
    [switch]$new
    [string]$queue_file
    [string]$vers = '2.4.8'
    [array]$running
    [array]$workers
    [string]$coderoot = ('\\' + $env:computername+'\'+$PSScriptRoot + '\..\..\..\..\AstroPathPipeline') -replace ":", "$"
    [PSCredential]$cred
    #
    Queue($mpath, $module, $cred){
        $this.module = $module 
        $this.mpath = $mpath
        $this.cred = $cred
        $this.Run()
    }
    #
    Run(){
        while(1){
            $this.ExtractQueue()
            $this.checknew()
            $this.InitializeWorkerlist()
            $this.GetRunningJobs()
            $this.DistributeTasks()
            $this.WaitTask()
        }
    }
    #
    # gets available tasks from the queue
    #
    ExtractQueue(){
        #
        $queue_path = $this.mpath + '\across_project_queues'
        $this.queue_file = $queue_path + '\' + $this.module + '-queue.csv'
        $queue_data = $this.getcontent($this.queue_file)
        #
        $current_queue_data = @()
        #
        # find rows without "processing started"
        #
        foreach($row in $queue_data) {
            $array = $row.ToString().Split(",")
            $array = $array -replace '\s',''
            if($array[3]){
                if($array -match "Processing"){ Continue } else { 
                    $current_queue_data += $row
                    }
                } 
        }
        #
        $this.tasks = $current_queue_data
        #
    }
    #
    # checks for new tasks to process
    #
    checknew(){
        #
        while (!($this.tasks)){
            Write-Host "No new samples to process." -ForegroundColor Yellow
            Start-Sleep -s (10 * 60)
            $this.tasks = $this.ExtractQueue()
        }
        #
    }
    #
    # get list of worker names \ locations
    #
    InitializeWorkerlist(){
        #
        # get list of Vitual machines that are on
        #
        if ($this.module -eq "inform"){
            #
            #$this.workers = invoke-command -ComputerName bki05 -Credential $this.cred `
            #     -ScriptBlock {(Get-VM | where {$_.State -eq 'RUNNING'}).Name}
            # $this.workers = (Get-VM -ComputerName bki05 -Credential $this.cred | where {$_.State -eq 'Running'}).Name
            #
            # should check current computer name and need to add a list of computers to use
            #
            $this.workers = (Get-VM | where {$_.State -eq 'Running'}).Name
            #
            # remove any of the "Taube Lab Workstations" from usable VMs (VMs 2)
            #
            [System.Collections.ArrayList]$TLWS = "VM_inForm_21","VM_inForm_22","VM_inForm_3"
            FOREACH ($WS in $TLWS){
                $CC = $WS + "$"
                $this.workers = $this.workers | Select-String $CC -notmatch
            }
        }
        #
        Write-Host "." -ForegroundColor Yellow
        Write-Host "Starting" $this.module "-Task-Distribution" -ForegroundColor Yellow
        write-host " Current Computers for Processing:" -ForegroundColor Yellow
        write-host " " $this.workers -ForegroundColor Yellow
        Write-Host "  ." -ForegroundColor Yellow
        #
        $this.workers = $this.workers
        #
    }
    #
    # get running jobs for this queue
    #
    GetRunningJobs(){
        #
        $this.running = @(Get-Job | Where-Object { $_.State -eq 'Running'  -and $_.Name -match 'inform'})
        if ($this.running){
            $locs = $this.running.Name.split('-')
            #
            FOREACH ($WS in $locs){
               $CC = $WS + "$"
               $this.workers = $this.workers | Select-String $CC -notmatch
            }
        }
        #
    }
    #
    DistributeTasks(){
        #
        # task object
        #
        $sb = {
                param(
                    $UserName, 
                    $Password,
                    $iworker,
                    $code,
                    $in,
                    $module,
                    $mpath
                )
            #
            psexec -i -u $UserName -p $Password \\$iworker cmd /c `
                "powershell -noprofile -executionpolicy bypass -command "" &{Import-Module $code; LaunchModule -module $module -mpath $mpath -in $in}"""
        }
        #
        # code to execute
        #
        $this.tasks1 = $this.tasks -replace ",,",","
        $this.tasks1 = $this.tasks1 -replace ",,",","
        $this.tasks1 = $this.tasks1 -replace ",,",","
        $this.tasks1 = $this.tasks1 -replace ",,",","
        $this.tasks1 = $this.tasks1 -replace ", ,",","
        $this.tasks1 = $this.tasks1 -replace ",  ,",","
        #
        While($this.workers -and $this.tasks){
            #
            # select the next usable VM
            #
            $cworker, $this.workers = $this.workers 
            $iworker = $cworker -replace '_',''
            $iworker = $iworker.ToLower()
            #
            # select the current task
            #
            $tasks_check, $this.tasks= $this.tasks
            $in, $this.tasks1 = $this.tasks1
            $in1 = "'"+$in+$this.vers + "'"
            #
            Write-Host "  Launching Task on:" $cworker -ForegroundColor Yellow
            #
            # launch the task
            #
            $UserName = $this.cred.UserName
            $Password = $this.cred.GetNetworkCredential().Password
            $nm = $cworker.tostring() + '-' + $this.module
            #
            start-job -ScriptBlock $sb -ArgumentList $UserName,$Password,$iworker,$this.coderoot,$in1,$this.module,$this.mpath -Name $nm
            #
            # Update the queue line
            #
            $this.UpdateQueue($in, $cworker, $tasks_check)
            #
        }
        #
    }
    #
    UpdateQueue ($in, $cVM, $CI_check){
        #
        $D = Get-Date
        $CI2 = "$in" + "Processing sent to: " + $cVM + " on: " + $D
        $mxtstring = 'Global\' + $this.queue_file.replace('\', '_') + '.LOCK'
        #
        #
        # add escape to '\'
        #
        $rg = [regex]::escape($CI_check) + "$"
        #
        $cnt = 0
        $Max = 120
        #
        do{
           $mxtx = New-Object System.Threading.Mutex($false, $mxtstring)
            try{
                $imxtx = $mxtx.WaitOne(60 * 10)
                if($imxtx){
                    $Q = get-content -Path $this.queue_file
                    $Q2 = $Q -replace $rg,$CI2
                    Set-Content -Path $this.queue_file -Value $Q2
                    $mxtx.releasemutex()
                    break
                } else{
                    $cnt = $cnt + 1
                    Start-Sleep -s 5
                }
            }catch{
                $cnt = $cnt + 1
                Start-Sleep -s 5
                Continue
            }
        } while($cnt -lt $Max)
        #
        # if the script could not access the queue file after 10 mins of trying every 2 secs
        # there is an issue and exit the script
        #
        if ($cnt -ge $Max){
            $ErrorMessage = "Could not access "+$this.module+"-queue.csv"
            Throw $ErrorMessage 
        }
        #
    }
    #
    WaitTask(){
        $run = @(Get-Job | Where-Object { $_.State -eq 'Running'  -and $_.Name -match 'inform'}).id
        if (!$this.workers -and $run){
            Wait-Job -id $run -Any
        }
    }
    #
}