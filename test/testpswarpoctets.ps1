<# -------------------------------------------
 testpswarpoctets
 created by: Andrew Jorquera
 Last Edit: 01.25.2022
 --------------------------------------------
 Description
 test if the methods of warpoctets are 
 functioning as intended
 -------------------------------------------#>
#
Class testpswarpoctets {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testpswarpoctets(){
        #
        # Setup Testing
        #
        $this.importmodule()
        #
        $task = ('0', 'M21_1', $this.process_loc, $this.mpath)
        $inp = warpoctets $task
        #
        # Run Tests
        #
        $this.DownloadFilesTest($inp)
        #$this.ShredDatTest($inp)
        $this.GetWarpOctetsTest($inp)
        $this.CleanupTest()
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
    }
    #
    [void]DownloadFilesTest($inp){
        Write-Host 'Starting Download Files Test'
        $inp.DownloadFiles()
        $im3path = $inp.processvars[2] + '/../Scan1/MSI/*.im3'
        if (!(@(Test-Path $im3path))) {
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
    [void]GetWarpOctetsTest($inp){
        Write-Host 'Starting GetWarpOctets Test'
        $output = $this.basepath + '\warping\octets\' + $this.slideid + '-all_overlap_octets.csv'
        $inp.warpoctets()
        if (!(test-path $output)){
            Throw 'Warp Octets Test Failed - Output file does not exist'
        }
        Write-Host 'Passed GetWarpOctets Test'
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
        Write-Host 'Processing Folder Deleted'
        Write-Host 'Passed Cleanup Test'
    }
    #
}
#
# launch test and exit if no error found
#
$test = [testpswarpoctets]::new()
exit 0