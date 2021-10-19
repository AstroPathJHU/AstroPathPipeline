<#
--------------------------------------------------------
meanimagecomparison
Created By: Andrew Jorquera
Last Edit: 10/13/2021
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
Function meanimagecomparison {
     #
     param($task, $log)
     #
     # used for testing; when launched manually without launchmodule
     #
     if (!($PSBoundParameters.ContainsKey('log'))){ 
        $log = [launchmodule]::new($task[$task.Count-1], 'meanimagecomparison', $task) 
        $e = 1
     } else {$e = 0}
     #
     $inp = [meanimagecomparison]::new($task, $log)
     if ($e -ne 1){
         $inp.RunMeanImageComparison()
     } else{
        return $inp
     }
     #
}
#