<# -------------------------------------------
 testsegmaps
 created by: Andrew Jorquera
 Last Edit: 01.24.2022
 --------------------------------------------
 Description
 test if the methods of segmaps are 
 functioning as intended
 -------------------------------------------#>
#
Class testsegmaps {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testsegmaps(){
        #
        # Setup Testing
        #
        $this.importmodule()
        #
        $task = ('0', 'M21_1', $this.process_loc, $this.mpath)
        $inp = segmaps $task
        #
        # Run Tests
        #
        $this.CleanupTest($inp)
        $this.GetaSegTest($inp)
        $this.GetnoSegTest($inp)
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
    }
    #
    [void]CleanupTest($inp){
        Write-Host 'Starting Cleanup Test'
        $inp.cleanup()
        if ($inp.processvars[4]) {
            $sor = $inp.sample.componentfolder()
            Write-Host 'Component Folder: ' $sor
            Write-Host 'Deleting these files: ' (gci -Path $sor -Include *w_seg.tif -Recurse)
            try {
                Get-ChildItem -Path $sor -Include *w_seg.tif -Recurse | Remove-Item -force
            }
            catch {
                Throw 'Error deleting segmentation files'
            }
        }
        Write-Host 'Passed Cleanup Test'
    }
    #
    [void]GetaSegTest($inp){
        Write-Host 'Starting GetaSeg Test'
        $table = $inp.sample.phenotypefolder() + '\Results\Tables'
        if (!(test-path $table + '\*csv')){
            Throw 'Phenotype Tables do not exist'
        }
        $inp.GetaSeg()
        $comp = (gci ($table + '\*') '*csv').Count
        $seg = (gci ($inp.sample.componentfolder() + '\*') '*data_w_seg.tif').Count
        if (!($comp -eq $seg)){
            Throw 'Component data count ~= Segmentation Data count'
        }
        Write-Host 'Passed GetaSeg Test'
    }
    #
    [void]GetnoSegTest($inp){
        Write-Host 'Starting GetnoSeg Test'
        try {
            $inp.GetnoSeg()
        }
        catch {
            Throw 'Error running GetnoSeg'
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