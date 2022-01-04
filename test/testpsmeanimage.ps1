<# -------------------------------------------
 testpsmeanimage
 created by: Andrew Jorquera
 Last Edit: 12/17/2021
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
        $datalocation = $PSScriptRoot + '/data'
        $task = ('1', 'M21_1', $processing, $datalocation)
        $inp = meanimage $task
        #
        # Run Tests
        #
        $this.DownloadFilesTest()
        #$this.ShredDatTest()
        #$this.ReturnDataTest()
        #$this.CleanupTest()
    }
    #
    [void]DownloadFilesTest(){
        $this.DownloadFiles()
        $xmlpath = $this.processvars[1] + '\' + $this.sample.slideid + '\*.xml'
        $im3path = $this.processvars[2] + '\..\Scan1\MSI\*.im3'
        if (!(@(Test-Path $xmlpath) -and @(Test-Path $im3path))) {
            Write-Error 'Download Files Test Failed'
            exit 1
        }
    }
    #
    [void]ShredDatTest(){
        $this.ShredDat()
        $datpath = $this.processvars[1] + '\' + $this.sample.slideid + '\*.dat'
        if (!(@(Test-Path $datpath))) {
            Write-Error 'Shred Dat Test Failed'
            exit 1
        }
    }
    #
    [void]ReturnDataTest(){
        $this.returndata()
        $returnpath = $this.sample.im3folder() + '\meanimage'
        if (!(@(Test-Path $returnpath))) {
            Write-Error 'Return Data Test Failed'
            exit 1
        }
    }
    #
    [void]CleanupTest(){
        $this.cleanup()
        if ($this.processvars[4]) {
            if (@(Test-Path $this.processvars[0])) {
                Write-Error 'Cleanup Test Failed'
                exit 1
            }
        }
    }
}
#
# launch test and exit if no error found
#
$test = [testpsmeanimage]::new()
exit 0

#Remove temporary processing directory
#$inp.sample.removedir($processing)