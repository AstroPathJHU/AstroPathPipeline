<# -------------------------------------------
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
        $mxtxid = 'Global\' + $module.replace('\', '_') + '.LOCK'
        $mxtx = New-Object System.Threading.Mutex -ArgumentList 'false', $mxtxid
        $mxtx.ReleaseMutex()
        if($error){
            Throw 'Module could not be imported'
        }
        $processing = $PSScriptRoot + '/test_for_jenkins/testing_meanimage'
        $datalocation = $PSScriptRoot + '/data'
        $task = ('1', 'M21_1', $processing, $datalocation)
        $inp = meanimage $task
        #
        # Run Tests
        #
        #$this.DownloadFilesTest()
        #$this.ShredDatTest()
        #$this.ReturnDataTest()
        #$this.CleanupTest()
    }
    #
    [void]DownloadFilesTest(){
        Write-Host 'Starting Download Files Test'
        $this.DownloadFiles()
        $xmlpath = $this.processvars[1] + '/' + $this.sample.slideid + '/*.xml'
        Write-Host 'xml path: ' $xmlpath
        $im3path = $this.processvars[2] + '/../Scan1/MSI/*.im3'
        if (!(@(Test-Path $xmlpath) -and @(Test-Path $im3path))) {
            Write-Error 'Download Files Test Failed'
            exit 1
        }
        Write-Host 'Passed Download Files Test'
    }
    #
    [void]ShredDatTest(){
        Write-Host 'Starting Shred Dat Test'
        $this.ShredDat()
        $datpath = $this.processvars[1] + '/' + $this.sample.slideid + '/*.dat'
        if (!(@(Test-Path $datpath))) {
            Write-Error 'Shred Dat Test Failed'
            exit 1
        }
        Write-Host 'Passed Shred Dat Test'
    }
    #
    [void]ReturnDataTest(){
        Write-Host 'Starting Return Data Test'
        $this.returndata()
        $returnpath = $this.sample.im3folder() + '\meanimage'
        if (!(@(Test-Path $returnpath))) {
            Write-Error 'Return Data Test Failed'
            exit 1
        }
        Write-Host 'Passed Return Data Test'
    }
    #
    [void]CleanupTest(){
        Write-Host 'Starting Cleanup Test'
        $this.cleanup()
        if ($this.processvars[4]) {
            if (@(Test-Path $this.processvars[0])) {
                Write-Error 'Cleanup Test Failed'
                exit 1
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