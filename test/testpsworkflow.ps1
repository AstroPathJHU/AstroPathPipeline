  
<# -------------------------------------------
 testpsdistpatcher
 created by: Benjamin Green
 Last Edit: 1/10/2022
 --------------------------------------------
 Description
 test if the dispatcher works
 -------------------------------------------#>
#
Class testpsworkflow {
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
        $password = ConvertTo-SecureString "MyPlainTextPassword" -AsPlainText -Force
        $cred = New-Object System.Management.Automation.PSCredential ("username", $password)  
        #
        $this.testconstructors($cred)
        #  
        $inp = astropathworkflow -Credential $cred -mpath $this.mpath -test
        $inp.workerloglocation = $PSScriptRoot + '\data\workflowlogs'
        $this.testdefworkerlist($inp)
        #
    }
    #
    [void]importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing'
    }
    #
    [void]testconstructors([PSCredential]$cred){
        #
        Write-Host '[astropathworkflow] construction tests started'
        #
        try {
            astropathworkflow -Credential $cred -test | Out-NULL
        } catch {
            Throw ('[astropathworkflow] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        #
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
        Write-Host 'Starting worker list tests'
        #
        $inp.defworkerlist()
        #
        if ($inp.workers.Status -match 'RUNNING'){
            Throw 'Some workers tagged as running when they are not'
        }
        #
        $this.StartTestJob()
        $inp.CheckOrphan()
        #
        $currentworker = $inp.workers[0]
        $jobname = $inp.defjobname($currentworker)
        #
        $j = get-job -Name $jobname
        if (!($j) -OR (!($inp.workers.Status -match 'RUNNING'))){
            Throw 'orphaned task monitor failed to launch'
        }
        #
        start-sleep -s (1*15)
        if ($j.State -match 'Completed'){
            Throw 'orphaned task monitor exited early'
        }
        #
        start-sleep -s (1*65)
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
        $currentworker = $inp.workers[0]
        $jobname = $inp.defjobname($currentworker)
        #
         $myscriptblock = {
                param( $workertasklog)
                    powershell -noprofile -executionpolicy bypass -command "Start-Sleep -s (1*60)" `
                            *>> $workertasklog
            }
        #
        $myparameters = @{
            ScriptBlock = $myscriptblock
            ArgumentList = $inp.workertasklog($jobname)
            name = ($jobname + '-test')
            }
        #
        Start-Job @myparameters
        #
        $j = get-job -Name ($jobname + '-test')
        if (!$j){
            Throw 'test job not launched'
        }
    }
    #
}
#
# launch test and exit if no error found
#
$test = [testpsworkflow]::new()
exit 0

#Remove temporary processing directory
#$inp.sample.removedir($processing)
