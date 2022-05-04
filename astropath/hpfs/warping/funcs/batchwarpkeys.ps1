<#
--------------------------------------------------------
batchwarpkeys
Benjamin Green
Last Edit: 02.16.2022
--------------------------------------------------------
Description
launch batch warp keys for the particular input.
if run without the log the module is run in 
test mode and returns a batchwarpkeys class rather
than running the module.
--------------------------------------------------------
Input:
$task[string]: the 4 part comma separated list of project, 
    batchid, worker location, mpath.
    E.g. @(7,08, \\bki08\e$, \\bki04\astropath_processing)
$log[logger]: a logging object for the batchwarpkeys 
module. 
--------------------------------------------------------
#>
#
Function batchwarpkeys {
    #
    param($task, $log, [Parameter()][switch]$test)
    #
    # used for testing; when launched manually without launchmodule
    #
    if (!($PSBoundParameters.ContainsKey('log')) -or $PSBoundParameters.test){ 
       $log = [launchmodule]::new($task.mpath, 'batchwarpkeys', $task) 
       $e = 1
    } else {$e = 0}
    #
    $inp = [batchwarpkeys]::new($task, $log)
    if ($e -ne 1){
        $inp.Runbatchwarpkeys()
    } else{
       return $inp
    }
    #
}
#