#Set up testing environment
$modulelocation = $PSScriptRoot + '\..\astropath'
Import-Module $modulelocation
$processing = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
$datalocation = $PSScriptRoot + '\data'
$task = ('1', 'M21_1', $processing, $datalocation)
$inp = meanimage $task

#Download Files test
$inp.DownloadFiles()
$xmlpath = $inp.processvars[1] + '\' + $inp.sample.slideid + '\*.xml'
$im3path = $inp.processvars[2] + '\..\Scan1\MSI\*.im3'
if (!(@(Test-Path $xmlpath) -and @(Test-Path $im3path))) {
    Write-Error 'Download Files Test Failed'
}

#Shred Data.dat files test
$inp.ShredDat()
$datpath = $inp.processvars[1] + '\' + $inp.sample.slideid + '\*.dat'
if (!(@(Test-Path $datpath))) {
    Write-Error 'Shred Dat Test Failed'
}

#Return Data test
$inp.returndata()
$returnpath = $inp.sample.im3folder() + '\meanimage'
if (!(@(Test-Path $meanimagedatpath))) {
    Write-Error 'Return Data Test Failed'
}

#Cleanup test
$inp.cleanup()
if ($inp.processvars[4]) {
    if (@(Test-Path $inp.processvars[0])) {
        Write-Error 'Cleanup Test Failed'
    }
}

#Remove temporary processing directory
$inp.sample.removedir($processing)