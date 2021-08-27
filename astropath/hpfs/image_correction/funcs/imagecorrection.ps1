<#
--------------------------------------------------------
imagecorrection
Created By: Benjamin Green
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
Function imagecorrection {
     #
     param($task, $log)
     #
     # used for testing; when launched manually without launchmodule
     #
     if (!($PSBoundParameters.ContainsKey('log'))){ 
        $log = [launchmodule]::new($task[1], '\\bki04\astropath_processing', 'imagecorrection', $task, 1) 
        $e = 1
     } else {$e = 0}
     #
     $inp = [imagecorrection]::new($task, $log)
     if ($e -ne 1){
         $inp.RunImageCorrection()
     } else{
        return $inp
     }
     #
}
#