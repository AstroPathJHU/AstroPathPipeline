<#
--------------------------------------------------------
initmodule
Benjamin Green, Andrew Jorquera
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
Input:
$in[string]: the 3 part comma separated list of project, 
    slideid, and worker location.
    E.g. "7,M18_1,location"
$vers[string]: The version number of inform to use 
    (must be after the PerkinElmer to Akoya name switch)
    E.g.: "2.4.8"
--------------------------------------------------------
#>
#
Function initmodule {
    #
    param($task, $log, $module, [Parameter()][switch]$test, [Parameter()][switch]$interactive)
    #
    if ($task.ContainsKey('tasklogfile')){
        updateprocessinglog -logfile $task.tasklogfile -jobname $task.jobname `
            -sample $log -lineoutput (
            'processname:', $log.processname, '- processid:', $log.processid -join ' ')
    }
    #
    # used for testing; when launched manually without launchmodule
    #
    if (!($PSBoundParameters.ContainsKey('log')) -or $PSBoundParameters.test){ 
        $log = [launchmodule]::new($task.mpath, $module, $task) 
       $e = 1
    } else {$e = 0}
    #
    $inp = New-Object $module -ArgumentList ($task, $log)
    if ($e -eq 1 -or $PSBoundParameters.interactive){
       Write-host '4'
        return $inp
    }
    #
    try {
        $inp.('run' + $module)()
    } catch {
        $inp.silentcleanup()
        Throw $_.Exception
    }
    <#
    if ($task.ContainsKey('tasklogfile')){
        updateprocessinglog -logfile $task.tasklogfile -jobname $task.jobname `
            -sample $log -lineoutput (
            #### processing statistics')
    }
    #>
}
#