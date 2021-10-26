##
# launch a queue for a provide module with a provided mpath and credentials
#
class Dispatcher : queue{
    #
    [switch]$new
    [array]$running
    [array]$workers
    [PSCredential]$cred
    [string]$workerloglocation = '\\'+$env:ComputerName+'\c$\users\public\astropath\'
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
    #
    [void]Run(){
        #
        while(1){
            $this.ExtractQueue()
            $this.checknew()
            $this.InitializeWorkerlist()
            $this.GetRunningJobs()
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
        Write-Host "Starting the AstroPath Pipeline." -ForegroundColor Yellow
        Write-Host ("Module: " + $this.module) -ForegroundColor Yellow
        Write-Host ("Username: " + $this.cred.UserName) -ForegroundColor Yellow
        $this.defCodeRoot()
        $this.initepy()
    }
    #
    # initializing the python environment
    # upgrade python package if needed.
    #
    [void]initepy(){
        Write-Host "Initializing\updating the conda envir." -ForegroundColor Yellow
        $this.checkconda()
        $this.checkpyapenvir('U')
    }
    #
    # checks for new tasks to process
    #
    [void]CheckNew(){
        #
        while (!($this.cleanedtasks)){
            Write-Host "No new samples to process." -ForegroundColor Yellow
            Start-Sleep -s (10 * 60)
            $this.ExtractQueue()
            $this.CheckCompletedWorkers()
        }
        #
    }
    #
    # get list of worker names \ locations
    #
    [void]InitializeWorkerlist(){
        #
        # get list of workers machines that are on
        #
        $this.workers = $this.OpenCSVFile($this.mpath+'\AstroPathHPFWLocs.csv') |
                             Where-Object {$_.module -eq $this.module}
        
        #
        Write-Host "." -ForegroundColor Yellow
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
    #
    # get running jobs for this queue
    #
    [void]GetRunningJobs(){
        #
        $this.running = @(Get-Job | Where-Object { $_.State -eq 'Running' -and $_.Name -match $this.module})
        if ($this.running){
            $this.running.Name | FOREACH {
               $CC = $_
               $this.workers = $this.workers | where-object {(($_.server, $_.location, $_.module) -join('-')) -ne  $CC}
            }
        }
        #
    }
    #
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
    [void]LaunchTask($currenttask, $currentworker){
        #       
        $username = $this.cred.UserName
        $password = $this.cred.GetNetworkCredential().Password
        $currentworkerip = $this.defcurrentworkerip($currentworker)
        $jobname = $this.defjobname($currentworker)
        $workertaskfile = $this.PrepareWorkerFiles($currenttask, $jobname, $currentworker)
        #
        if ($currentworker.location -match 'VM'){
            $myscriptblock = {
                param($username, $password, $currentworkerip, $workertaskfile)
                psexec -i -nobanner -accepteula -u $username -p $password \\$currentworkerip `
                    powershell -noprofile -executionpolicy bypass -command "$workertaskfile" 
            }
        } else {
            $myscriptblock = {
                param($username, $password, $currentworkerip, $workertaskfile)
                psexec -nobanner -accepteula -u $username -p $password \\$currentworkerip `
                    powershell -noprofile -executionpolicy bypass -command "$workertaskfile" 
            }
        }
        #
        $myparameters = @{
            ScriptBlock = $myscriptblock
            ArgumentList = $username, $password, $currentworkerip, $workertaskfile
            name = $jobname
            }
        Start-Job @myparameters
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
        $currenttasktowrite = ("&{Import-Module ", $this.coderoot, ";LaunchModule -mpath:", `
                             $this.mpath, " -module:", $this.module, " -stringin:", `
                             $currenttaskinput, " } *>> '", $workerlogfile, "'") -join ''
        #
        $this.SetFile($workertaskfile, $currenttasktowrite)
        return $workertaskfile
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
            $donejobs | ForEach {Get-Content ($this.workerloglocation+$_.Name+'.log')}
            $donejobs | ForEach {Remove-Item ($this.workerloglocation+$_.Name+'.log') -force -ea SilentlyContinue}
        }
        #
    }
    #
}