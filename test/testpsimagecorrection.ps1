<# -------------------------------------------
 testpsimagecorrection
 created by: Andrew Jorquera
 Last Edit: 2/1/2022
 --------------------------------------------
 Description
 test if the methods of imagecorrection are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsimagecorrection {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testpsimagecorrection(){
        #
        # Setup Testing
        #
        $this.importmodule()
        #
        $task = ('0', 'M21_1', $this.process_loc, $this.mpath)
        $inp = imagecorrection $task
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
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_imagecorrection'
    }
    #
    [void]DownloadFilesTest($inp){
        #
        Write-Host 'Starting Download Files Test'
        $im3path = $inp.sample.basepath + '\' + $inp.sample.slideid + '\im3\Scan1\MSI'
        Write-Host 'im3path: ' $im3path
        Write-Host 'MSI folder: ' $inp.sample.MSIfolder()
        if (!([regex]::Escape($inp.sample.MSIfolder()) -contains [regex]::Escape($im3path))){
            Throw ('MSI folder not correct: ' + $inp.sample.MSIfolder() + '~=' + $im3path)
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
            Throw ('XML folder not correct: ' + $inp.sample.xmlfolder() + '~=' + $xmlpath)
        }
        $xmlpath += '\*xml'
        if (!(Test-Path -Path $xmlpath)) {
            Throw 'No xml files in MSI folder'
        }
        Write-Host 'Correct files in XML folder'
        #
        $testid = '08'
        Write-Host 'Batch ID: ' $inp.sample.BatchID
        if (!($inp.sample.BatchID -eq $testid)) {
            Throw 'Incorrect BatchID'
        }
        Write-Host 'Received Correct Batch ID'
        #
        Write-Host 'Flatfield Folder: ' $inp.sample.flatfieldfolder()
        Write-Host 'Passed Download Files Test'
        #
    }
    #
    [void]CleanupTest($inp){
        Write-Host 'Starting Cleanup Test'
        $flatwim3path = $inp.sample.basepath + '\' + $inp.sample.slideid + '\im3'
        New-Item -Path $flatwim3path -Name "flatw" -ItemType "directory"
        $flatwim3path += '\flatw'
        Write-Host 'flatwim3path: ' $flatwim3path
        Write-Host 'Flatw IM3 folder: ' $inp.sample.flatwim3folder()
        if (!([regex]::Escape($inp.sample.flatwim3folder()) -contains [regex]::Escape($flatwim3path))){
            Throw ('MSI folder not correct: ' + $inp.sample.flatwim3folder() + '~=' + $flatwim3path)
        }
        Write-Host 'Recieved Correct FlatwIM3 Folder'
        #
        Write-Host 'Flatw Folder: ' $inp.sample.flatwfolder()
        #
        $inp.sample.removedir($this.processloc)
        #
        Write-Host 'Passed Cleanup Test'
    }
}
#
# launch test and exit if no error found
#
try { 
    [testpsimagecorrection]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0v