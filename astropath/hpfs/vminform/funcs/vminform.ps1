<#
--------------------------------------------------------
inform_worker
Created By: Benjamin Green
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
Input:
$in[string]: the 4 part comma separated list of dpath, 
    slideid, antibody, and algorithm.
    E.g. "\\bki04\Clinical_Specimen_2,M18_1,CD8,CD8_12.05.2018_highTH.ifr"
$vers[string]: The version number of inform to use 
    (must be after the PerkinElmer to Akoya name switch)
    E.g.: "2.4.8"
--------------------------------------------------------
#>
#
Function vminform {
     #
     param($task, $log)
     #
     # used for testing; when launched manually without launchmodule
     #
     if (!($PSBoundParameters.ContainsKey('log'))){ 
        $log = [launchmodule]::new('\\bki04\astropath_processing', 'vminform', $task) 
        $e = 1
     } else {$e = 0}
     #
     $inp = [informinput]::new($task, $log)
     if ($e -ne 1){
         $inp.RunBatchInForm()
     } else{
        return $inp
     }
     #
}
#