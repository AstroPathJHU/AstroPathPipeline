﻿<#
--------------------------------------------------------
batchflatfield
Created By: Andrew Jorquera
Last Edit: 09/23/2021
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
Function batchflatfield {
     #
     param($task, $log, [Parameter()][switch]$test)
     #
     # used for testing; when launched manually without launchmodule
     #
     if (!($PSBoundParameters.ContainsKey('log')) -or $PSBoundParameters.test){ 
        $log = [launchmodule]::new($task.mpath, 'batchflatfield', $task) 
        $e = 1
     } else {$e = 0}
     #
     $inp = [batchflatfield]::new($task, $log)
     if ($e -ne 1){
         $inp.RunBatchFlatfield()
     } else{
        return $inp
     }
     #
}
#