<# -------------------------------------------
 testpsmeanimage
 created by: Andrew Jorquera
 Last Edit: 01.24.2022
 --------------------------------------------
 Description
 test if the methods of segmaps are 
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
        $inp = segmaps $task
        #
        Write-Host $inp
        #
        # Run Tests
        #
        $this.CleanupTest($inp)
        #$this.GetaSegTest($inp)
        #$this.GetnoSegTest($inp)
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
    }
    [void]CleanupTest($inp){
        Write-Host 'Starting Cleanup Test'
        #$inp.cleanup()
        if ($inp.processvars[4]) {
            $sor = $inp.sample.componentfolder()
            #Get-ChildItem -Path $sor -Include *w_seg.tif -Recurse | Remove-Item -force
            Write-Host 'Component Folder: ' $sor
            Write-Host 'All Files: ' (gci -Path $sor)
            Write-Host 'Deleting these files: ' (gci -Path $sor -Include *w_seg.tif -Recurse)

            #if (@(Test-Path $inp.processvars[0])) {
            #    Throw 'Cleanup Test Failed'
            #}
        }
        Write-Host 'Passed Cleanup Test'
    }
    #
    [void]GetaSegTest($inp){
        Write-Host 'Starting GetaSeg Test'
        $inp.GetaSeg()
        $xmlpath = $inp.processvars[1] + '/' + $inp.sample.slideid + '/*.xml'
        Write-Host 'xml path: ' $xmlpath
        $im3path = $inp.processvars[2] + '/../Scan1/MSI/*.im3'
        if (!(@(Test-Path $xmlpath) -and @(Test-Path $im3path))) {
            Throw 'GetaSeg Failed'
        }
        Write-Host 'Passed GetaSeg Test'
    }
    #
    [void]GetnoSegTest($inp){
        Write-Host 'Starting GetnoSeg Test'
        $inp.GetnoSeg()
        $datpath = $inp.processvars[1] + '/' + $inp.sample.slideid + '/*.dat'
        if (!(@(Test-Path $datpath))) {
            Throw 'GetnoSeg Test Failed'
        }
        Write-Host 'Passed GetnoSeg Test'
    }
    #
}
#
# launch test and exit if no error found
#
$test = [testsegmaps]::new()
exit 0