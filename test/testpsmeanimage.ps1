﻿<# -------------------------------------------
 testpsmeanimage
 created by: Andrew Jorquera
 Last Edit: 1/5/2021
 --------------------------------------------
 Description
 test if the methods of meanimage are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsmeanimage {
    #
    testpsmeanimage(){
        #
        # Setup Testing
        #
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        if($error){
            Throw 'Module could not be imported'
        }
        $processing = $PSScriptRoot + '/test_for_jenkins/testing_meanimage'
        $datalocation = $PSScriptRoot + '/data/astropath_processing'
        #
        Write-Host 'DataLocation: ' $datalocation
        #New-Item C:\Users\ajorque1\Desktop\Test.csv -value ("Project,Dpath,Dname,Spath,Cpath,FWpath`n0," + '\data' + ",astropath_processing,tme1\Vectra3,bki03\Compressed_Clinical_Specimens,bki08\h$\flatw_testing")

        
        #
        $task = ('1', 'M21_1', $processing, $datalocation)
        $inp = meanimage $task
        #
        # Run Tests
        #
        #$this.DownloadFilesTest($inp)
        #$this.ShredDatTest($inp)
        #$this.ReturnDataTest($inp)
        #$this.CleanupTest($inp)
    }
    #
    [void]DownloadFilesTest($inp){
        Write-Host 'Starting Download Files Test'
        $inp.DownloadFiles()
        $xmlpath = $inp.processvars[1] + '/' + $inp.sample.slideid + '/*.xml'
        Write-Host 'xml path: ' $xmlpath
        $im3path = $inp.processvars[2] + '/../Scan1/MSI/*.im3'
        if (!(@(Test-Path $xmlpath) -and @(Test-Path $im3path))) {
            Throw 'Download Files Test Failed'
        }
        Write-Host 'Passed Download Files Test'
    }
    #
    [void]ShredDatTest($inp){
        Write-Host 'Starting Shred Dat Test'
        $inp.ShredDat()
        $datpath = $inp.processvars[1] + '/' + $inp.sample.slideid + '/*.dat'
        if (!(@(Test-Path $datpath))) {
            Throw 'Shred Dat Test Failed'
        }
        Write-Host 'Passed Shred Dat Test'
    }
    #
    [void]ReturnDataTest($inp){
        Write-Host 'Starting Return Data Test'
        $inp.returndata()
        $returnpath = $inp.sample.im3folder() + '\meanimage'
        if (!(@(Test-Path $returnpath))) {
            Throw 'Return Data Test Failed'
        }
        Write-Host 'Passed Return Data Test'
    }
    #
    [void]CleanupTest($inp){
        Write-Host 'Starting Cleanup Test'
        $inp.cleanup()
        if ($inp.processvars[4]) {
            if (@(Test-Path $inp.processvars[0])) {
                Throw 'Cleanup Test Failed'
            }
        }
        Write-Host 'Passed Cleanup Test'
    }
}
#
# launch test and exit if no error found
#
$test = [testpsmeanimage]::new()
exit 0

#Remove temporary processing directory
#$inp.sample.removedir($processing)