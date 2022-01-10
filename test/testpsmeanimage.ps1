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
    [meanimage]$inp
    #
    testpsmeanimage(){
        #
        # Setup Testing
        #
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        try {
            #$mxtxid = 'Global\' + $module.replace('\', '_') + '.LOCK'
            $mxtxid = 'TestLock.LOCK'
            $mxtx = New-Object System.Threading.Mutex -ArgumentList 'false',$mxtxid
            $mxtx.ReleaseMutex()
        } catch {
            Write-Host 'Mutex Error'
            exit 1
        }
        if($error){
            Throw 'Module could not be imported'
        }
        $processing = $PSScriptRoot + '/test_for_jenkins/testing_meanimage'
        $datalocation = $PSScriptRoot + '/data'
        $task = ('1', 'M21_1', $processing, $datalocation)
        $this.inp = meanimage $task
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
        $this.inp.DownloadFiles()
        $xmlpath = $this.inp.processvars[1] + '/' + $this.inp.sample.slideid + '/*.xml'
        Write-Host 'xml path: ' $xmlpath
        $im3path = $this.inp.processvars[2] + '/../Scan1/MSI/*.im3'
        if (!(@(Test-Path $xmlpath) -and @(Test-Path $im3path))) {
            Throw 'Download Files Test Failed'
        }
        Write-Host 'Passed Download Files Test'
    }
    #
    [void]ShredDatTest(){
        Write-Host 'Starting Shred Dat Test'
        $this.inp.ShredDat()
        $datpath = $this.inp.processvars[1] + '/' + $this.inp.sample.slideid + '/*.dat'
        if (!(@(Test-Path $datpath))) {
            Throw 'Shred Dat Test Failed'
        }
        Write-Host 'Passed Shred Dat Test'
    }
    #
    [void]ReturnDataTest(){
        Write-Host 'Starting Return Data Test'
        $this.inp.returndata()
        $returnpath = $this.inp.sample.im3folder() + '\meanimage'
        if (!(@(Test-Path $returnpath))) {
            Throw 'Return Data Test Failed'
        }
        Write-Host 'Passed Return Data Test'
    }
    #
    [void]CleanupTest(){
        Write-Host 'Starting Cleanup Test'
        $this.inp.cleanup()
        if ($this.inp.processvars[4]) {
            if (@(Test-Path $this.inp.processvars[0])) {
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