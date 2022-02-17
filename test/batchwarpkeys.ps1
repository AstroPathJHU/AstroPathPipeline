#
$apmodule = $PSScriptRoot + '/../astropath'
Import-Module $apmodule
#
$project = '01'
$batchid = '01'
$processloc = '\\bki08\E$'
$mpath = '\\bki04\astropath_processing'
$task = ($project, $batchid, $processloc, $mpath)
$inp = batchwarpkeys $task
#
$inp.runbatchwarpkeys()