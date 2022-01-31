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
        $this.CleanupTest($inp)
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_warpoctets'
    }
    #
    [void]DownloadFilesTest($inp){
        #
        Write-Host 'Starting Download Files Test'
        $im3path = $inp.sample.basepath + '\' + $inp.sample.slideid + '\im3\Scan1\MSI'
        Write-Host 'im3path: ' $im3path
        Write-Host 'MSI folder: ' $inp.sample.MSIfolder()
        if (!([regex]::Escape($inp.sample.MSIfolder()) -contains [regex]::Escape($im3path))){
            Throw ('MSI folder not correct: ' + $inp.MSIfolder() + '~=' + $im3path)
        }
        $im3path += '\*im3'
        if (!(Test-Path -Path $im3path)) {
            Throw 'No im3 files in MSI folder'
        }
        Write-Host 'Correct files in IM3 folder'
        Write-Host 'Passed Download Files Test'
        #
    }
    #
    [void]CleanupTest($inp){
        Write-Host 'Starting Cleanup Test'
        if ($inp.processvars[4]) {
            $inp.cleanup()
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