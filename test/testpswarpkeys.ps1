#
Write-Host 'importing module'
#
$apmodule = $PSScriptRoot + '/../astropath'
Import-Module $apmodule
#
$project = '1'
$batchid = '6'
$processloc = '\\bki08\E$'
$mpath = '\\bki04\astropath_processing'
$taskname = 'warpingcohort'
#
Write-Host 'building batchwarpkeys task object'
$task = ($project, $batchid, $processloc, $mpath)
$inp = batchwarpkeys $task
#
Write-Host $inp.processloc
$inp.sample.createdirs($inp.processloc)
#
Write-Host $inp.sample.slideid
#
$inp.getmodulename()
$dpath = $inp.sample.basepath
$rpath = '\\' + $inp.sample.project_data.fwpath
#
Write-Host $inp.sample.pybatchflatfieldfullpath()
if (!(Test-Path $inp.sample.pybatchflatfieldfullpath())){
    Throw ('flatfield file does not exist: ' + $inp.sample.pybatchflatfieldfullpath())
}
#
$pythontask = $inp.getpythontask($dpath, $rpath)
#
# $batchslides = $inp.sample.batchslides.slideid -join '|'
# $pythontask = $inp.getpythontask($dpath, $rpath, $batchslides)
#
Write-Host 'running: '
Write-Host $pythontask
#
$inp.sample.checkconda()
conda activate $inp.sample.pyenv()
Invoke-Expression $pythontask
conda deactivate 
#        
#$inp.runpythontask($taskname, $pythontask)
#
#$inp.runbatchwarpkeys()

