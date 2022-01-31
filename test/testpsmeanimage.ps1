<# -------------------------------------------
 testpsmeanimage
 created by: Andrew Jorquera
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the methods of meanimage are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsmeanimage {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testpsmeanimage(){
        #
        # Setup Testing
        #
        $this.importmodule()
        #
        $task = ('0', 'M21_1', $this.process_loc, $this.mpath)
        $inp = meanimage $task
        #
        # Run Tests
        #
        $this.DownloadFilesTest($inp)
        $this.ReturnDataTest($inp)
        $this.CleanupTest($inp)
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
        #
        $xmlpath = $inp.sample.basepath + '\' + $inp.sample.slideid + '\im3\xml'
        Write-Host 'xmlpath: ' $xmlpath
        Write-Host 'XML folder: ' $inp.sample.xmlfolder()
        if (!([regex]::Escape($inp.sample.xmlfolder()) -contains [regex]::Escape($xmlpath))){
            Throw ('XML folder not correct: ' + $inp.xmlfolder() + '~=' + $xmlpath)
        }
        $xmlpath += '\*xml'
        if (!(Test-Path -Path $xmlpath)) {
            Throw 'No xml files in MSI folder'
        }
        Write-Host 'Correct files in XML folder'
        Write-Host 'Passed Download Files Test'
        #
    }
    #
    [void]ReturnDataTest($inp){
        Write-Host 'Starting Return Data Test'
        $sourcepath = $inp.processvars[0]
        $returnpath = $inp.sample.im3folder() + '\meanimage'
        Write-Host 'Source Path: ' $sourcepath
        Write-Host 'Return Path: ' $returnpath
        #
        if ($inp.processvars[4]) {
            #
            New-Item -Path $sourcepath -Name "meanimage" -ItemType "directory"
            if (!(@(Test-Path $sourcepath))) {
                Throw 'Return Data Test Failed - Source path does not exist'
            }
            #
            if (!(@(Test-Path $returnpath))) {
                Throw 'Return Data Test Failed - Return path does not exist'
            }
        }
        Write-Host 'Passed Return Data Test'
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
}
#
# launch test and exit if no error found
#
$test = [testpsmeanimage]::new()
exit 0