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
        $this.SegMapsTest($inp)
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_segmaps'
    }
    #
    [void]CleanupTest($inp){
        Write-Host 'Starting Cleanup Test'
        $sor = $inp.sample.componentfolder()
        Write-Host 'Component Folder: ' $sor
        $seg_files = (gci -Path $sor -Include *w_seg.tif -Recurse)
        Write-Host 'Deleting component data with segmentation data: ' $seg_files
        $inp.sample.copy($sor, $this.process_loc, 'W_seg.tif', 8)
        #
        $inp.cleanup()
        $seg_files = (gci -Path $sor -Include *w_seg.tif -Recurse)
        if (!($seg_files.Count -eq 0)) {
            Throw 'Error deleting component data with segmentation data'
        }
        $inp.sample.copy($this.process_loc, $sor, 'w_seg.tif', 8)
        Write-Host 'Files at End: ' (gci -Path $sor -Recurse)
        Write-Host 'Passed Cleanup Test'
    }
    #
    [void]SegMapsTest($inp){
        Write-Host 'Starting SegMaps Test'
        $table = $inp.sample.phenotypefolder() + '\Results\Tables'
        if (!(test-path ($table + '\*csv'))){
            Throw 'Phenotype Tables do not exist'
        }
        Write-Host 'Phenotype tables checked'
        $comp = (gci ($table + '\*') '*csv').Count
        $seg = (gci ($inp.sample.componentfolder() + '\*') '*data_w_seg.tif').Count
        if (!($comp -eq $seg)){
            Throw 'Component data count ~= Segmentation Data count'
        }
        Write-Host 'Passed SegMaps Test'
    }
    #
}
#
# launch test and exit if no error found
#
$test = [testsegmaps]::new()
exit 0