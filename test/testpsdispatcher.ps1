  
<# -------------------------------------------
 testpsdistpatcher
 created by: Benjamin Green
 Last Edit: 1/10/2022
 --------------------------------------------
 Description
 test if the dispatcher works
 -------------------------------------------#>
#
Class testpsdispatcher {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testpsdispatcher(){
        #
        # Setup Testing
        #
        $this.importmodule()
        #
        $cred = Get-Credential -Message "Provide a user name (domain\username) and password"
        $inp = [dispatcher]::new($this.mpath, 'shredxml', 'NA', 'NA', 'NA', $cred)
        #
    }
    #
    [void]importmodule(){
        #$module = '\\bki08\e$\working_code\dev\AstroPathPipelinePrivate\astropath'
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
    }
    #
    [void]TestIntializeWorkerList($inp){
        Write-Host 'Starting worker list tests'
        #
        $inp.InitializeWorkerlist()
        #
        if(!(($inp.workers.module | Sort-Object | Get-Unique) -contains 'shredxml')){
            Throw 'Work List not appropriately defined'
        }
        #
        $inp.GetRunningJobs()
        if ($inp.workers.count -ne 4){
            Throw 'Some workers tagged as running when they are not'
        }
        #
        $this.StartTestJob()
        $inp.GetRunningJobs()
        #
        $currentworker = $inp.workers[0]
        $jobname = $inp.defjobname($currentworker)
        #
        $j = get-job -Name $jobname
        if (!($j)){
            Throw 'orphaned task monitor failed to launch'
        }
        #
        start-sleep -s (1*65)
        #
        if(!((get-job -Name $jobname).State -match 'Completed')){
             Throw 'orphaned task monitor did not close correctly'
        }
        #
        Write-Host 'Passed worker list tests'
        #
    }
    #
    [void]StartTestJob($inp){
        $currentworker = $inp.workers[0]
        $creds = $inp.GetCreds()  
        $currentworkerip = $inp.defcurrentworkerip($currentworker)
        $jobname = $inp.defjobname($currentworker)

         $myscriptblock = {
                param($username, $password, $currentworkerip, $workertasklog)
                psexec -i -nobanner -accepteula -u $username -p $password \\$currentworkerip `
                    powershell -noprofile -executionpolicy bypass -command "Start-Sleep -s (1*60)" `
                    *>> $workertasklog
            }
        #
        $myparameters = @{
            ScriptBlock = $myscriptblock
            ArgumentList = $creds[0], $creds[1], $currentworkerip, $inp.workertasklog($jobname)
            name = ($jobname + '-test')
            }
        #
        Start-Job @myparameters
    }
    #
}
#
# launch test and exit if no error found
#
$test = [testpsdispatcher]::new()
exit 0

#Remove temporary processing directory
#$inp.sample.removedir($processing)
