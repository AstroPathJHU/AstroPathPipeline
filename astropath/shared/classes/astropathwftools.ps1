#
#
#
class astropathwftools : sampledb {
    #
    [Pscredential]$login
    [PSCustomObject]$workers
    [string]$workerloglocation = '\\' + $env:ComputerName +
        '\c$\users\public\astropath\'   
    #
    astropathwftools(){
        $this.init("NA", '')
    }
    #
    astropathwftools($login){
        $this.init($login, '')
    }
    #
    astropathwftools($login, $mpath) : base($mpath) {
        $this.init($login, '')
    }
    #
    astropathwftools($login, $mpath, $projects) : base($mpath, $projects){
        $this.init($login, '')
    }
        #
    astropathwftools($login, $mpath, $projects, $modules) : base($mpath, $projects){
        $this.init($login, $modules)
    }
    #
    # initialize the code
    #
    [void]init($login, $modules){
        #
        $this.login = $login
        if ($modules){
            $this.modules = $modules
        } else {
            $this.getmodulenames()
        }
        #
        Write-Host "Starting the AstroPath Pipeline" -ForegroundColor Yellow
        Write-Host ("Module: " + $this.module) -ForegroundColor Yellow
        Write-Host ("Username: " + $this.login.UserName) -ForegroundColor Yellow
        Write-Host ("Modules to launch: " + $this.modules) -ForegroundColor Yellow
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
        Write-Host " Starting-Task-Distribution" -ForegroundColor Yellow
        Write-Host " Finding Workers" -ForegroundColor Yellow
        $this.workers = $this.OpenCSVFile($this.mpath+'\AstroPathHPFWLocs.csv')
        $this.workers | Add-Member -NotePropertyName 'Status' -NotePropertyValue 'IDLE'
        $this.CheckOrphan()
        #
        write-host " Current Workers for Processing:" -ForegroundColor Yellow
        <#write-host " " ($this.workers | 
            Format-Table  @{Name="module";Expression = { $_.module }; Alignment="center" },
                            @{Name="server";Expression = { $_.server }; Alignment="center" },
                            @{Name="location";Expression = { $_.location }; Alignment="center" } |
            Out-String).Trim() -ForegroundColor Yellow
        #>
        Write-Output $this.workers
        Write-Host "  ." -ForegroundColor Yellow    
        #
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
        $this.workers | foreach-object{
            #
            $currentjob = $_
            $jobname = $this.defjobname($currentjob)
            $workertasklog = $this.workertasklog($jobname)
            $workertaskfile = $this.workertaskfile($jobname)
            #
            if (test-path $workertasklog){
                #
                $fileInfo = New-Object System.IO.FileInfo $workertasklog
                #
                Write-Host ('  Orphaned job found: ' + $workertasklog) `
                    -ForegroundColor Yellow
                #
                try {
                    $fileStream = $fileInfo.Open([System.IO.FileMode]::Open)
                    $fileStream.Dispose()
                    $this.removefile($workertasklog)
                    $this.CheckTaskLog($jobname, 'ERROR')
                    $this.removefile($workertaskfile)
                    Write-Host ('  Orphaned job completed and cleared: ' + $workertasklog) `
                        -ForegroundColor Yellow
                }catch {
                    $this.StartOrphanMonitor($jobname)
                    $_.Status = 'RUNNING'
                    Write-Host ('  Orphaned job not completed: ' + $workertasklog + '...watching log') `
                        -ForegroundColor Yellow
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
            while (1) {
                try { 
                    $fileInfo = New-Object System.IO.FileInfo $workertasklog
                    $fileStream = $fileInfo.Open([System.IO.FileMode]::Open)
                    $fileStream.Dispose()
                    break
                } catch {
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
                    wait-event $SI  
                    Unregister-Event -SourceIdentifier $SI -Force 
                    #
                }
            }
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
    #
    [string]DefJobName($currentworker){
        $jobname = ($currentworker.server, $currentworker.location, $currentworker.module) -join '-'     
        return $jobname
    }
    #
    [string]workertasklog($jobname){
        return $this.workerloglocation+$jobname+'-taskfile-job.log'
    }
    #
    [string]workertaskfile($jobname){
        return $this.workerloglocation+$jobname+'-taskfile.ps1'
    }
    #
    [string]workerlogfile($jobname){
        return $this.workerloglocation+$jobname+'.log'
    }
    [void]deftaskqueues(){
        $this.modules | ForEach-Object{

        }
    }
}