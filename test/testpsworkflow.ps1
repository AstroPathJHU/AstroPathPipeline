  
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
        #$password = ConvertTo-SecureString "MyPlainTextPassword" -AsPlainText -Force
        #$cred = New-Object System.Management.Automation.PSCredential ("username", $password)  
        $cred = [System.Net.CredentialCache]::DefaultNetworkCredentials
        #
        $this.testconstructors($cred)
        #  
        $inp = astropathworkflow -Credential $cred -mpath $this.mpath -test
        $inp.workerloglocation = $PSScriptRoot + '\data\workflowlogs\'
        $inp.createdirs($inp.workerloglocation)
        $this.testdefworkerlist($inp)
        $inp.removedir($inp.workerloglocation)
        Write-Host '.'
        Throw 'stop'
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
    [void]Testdefworkerlist($inp){
        #
        Write-Host "."
        Write-Host 'Starting worker list tests'
        #
        Write-Host "    Defining worker list"
        #
        $inp.importworkerlist($inp.mpath)
        $inp.CheckOrphan()
        #
        if ($inp.worker_data.Status -match 'RUNNING'){
            Throw 'Some workers tagged as running when they are not'
        }
        #
        Write-Host '    create a test job'
        #
        $this.StartTestJob($inp)
        #
        Write-Host '    launch orphan monitor for test job'
        #
        $inp.CheckOrphan()
        #
        $currentworker = $inp.workers[0]
        $jobname = $inp.defjobname($currentworker)
        #
        $j = get-job -Name $jobname
        #
        Write-Host '    job name:' $jobname
        #
        if (!($j) -OR (!($inp.workers.Status -match 'RUNNING'))){
            Throw 'orphaned task monitor failed to launch'
        }
        #
        start-sleep -s (1*15)
        if ($j.State -match 'Completed'){
            Throw 'orphaned task monitor exited early'
        }
        #
        $testj = get-job -Name ($jobname + '-test')
        wait-job $testj -timeout 180
        #
        write-host '    job state:' $j.State
        #
        if(!($j.State -match 'Completed')){
             Throw 'orphaned task monitor did not close correctly'
        }
        #
        Receive-Job $j -ErrorAction Stop
        #
        Write-Host 'Passed worker list tests'
        #
    }
    #
    [void]StartTestJob($inp){
        Write-Host '    Starting test job'
        #
        $currentworker = $inp.worker_data[0]
        $jobname = $inp.defjobname($currentworker)
        #
         $sb = {
                param($workertasklog)
                pwsh -noprofile -executionpolicy bypass -command `
                    "&{Write-Host 'Launched'; Start-Sleep -s (1*60)}" *>> $workertasklog
            }
        #
        $myparameters = @{
            ScriptBlock = $sb
            ArgumentList = $inp.workertasklog($jobname)
            name = ($jobname + '-test')
            }
        #
        Start-Job @myparameters
        #
        if ((get-job).Name -notcontains ($jobname + '-test')){
            Throw 'test job not launched'
        }
        #
        $j = get-job -Name ($jobname + '-test')
        Wait-Job $j -Timeout 5
        if ($j.State -notmatch 'Running'){
            Receive-Job $j -ErrorAction Stop
        }
        #
        if (!(test-path $inp.workertasklog($jobname))){
            Throw ('log not created: ' + $inp.workertasklog($jobname))
        }
        #
        Write-Host '    Test job launched'
    }
    #
}
#
# launch test and exit if no error found
#
try {
    [testpsworkflow]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0
