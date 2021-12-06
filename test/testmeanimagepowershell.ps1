﻿Write-Host 'Starting Tests... Script Root =' $PSScriptRoot
$modulelocation = $PSScriptRoot + '\..\astropath'
Import-Module $modulelocation
$processing = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
$datalocation = $PSScriptRoot + '\data'
Write-Host ' Processing Location =' $PSScriptRoot
Write-Host 'Data Location =' $PSScriptRoot
$task = ('1', 'M21_1', $processing, $datalocation)
$inp = meanimage $task

Write-Output $inp
exit 0

#Download Files test
$inp.DownloadFiles()
$xmlpath = $inp.processvars[1] + '\' + $inp.sample.slideid + '\*.xml'
$im3path = $inp.processvars[2] + '\..\Scan1\MSI\*.im3'
if (!(@(Test-Path $xmlpath) -and @(Test-Path $im3path))) {
    Write-Error 'Download Files Test Failed'
    exit 1
}

#Shred Data.dat files test
$inp.ShredDat()
$datpath = $inp.processvars[1] + '\' + $inp.sample.slideid + '\*.dat'
if (!(@(Test-Path $datpath))) {
    Write-Error 'Shred Dat Test Failed'
    exit 1
}

#Return Data test
$inp.returndata()
$returnpath = $inp.sample.im3folder() + '\meanimage'
if (!(@(Test-Path $meanimagedatpath))) {
    Write-Error 'Return Data Test Failed'
    exit 1
}

#Cleanup test
$inp.cleanup()
if ($inp.processvars[4]) {
    if (@(Test-Path $inp.processvars[0])) {
        Write-Error 'Cleanup Test Failed'
        exit 1
    }
}

#Remove temporary processing directory
$inp.sample.removedir($processing)

#Exit
exit 0