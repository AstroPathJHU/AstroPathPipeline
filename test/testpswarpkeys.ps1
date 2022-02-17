#
$apmodule = $PSScriptRoot + '/../astropath'
Import-Module $apmodule
#
$project = '1'
$batchid = '6'
$processloc = '\\bki08\E$'
$mpath = '\\bki04\astropath_processing'
$task = ($project, $batchid, $processloc, $mpath)
$inp = batchwarpkeys $task
#
$inp.getmodulename()
$dpath = $inp.sample.basepath
$rpath = '\\' + $inp.sample.project_data.fwpath
Write-Host $inp.getpythontask($dpath, $rpath)
#
$inp.runbatchwarpkeys()