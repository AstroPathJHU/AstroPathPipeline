﻿using module .\testtools.psm1
<# -------------------------------------------
 testpssegmaps
 Benjamin Green, Andrew Jorquera
 Last Edit: 01.28.2022
 --------------------------------------------
 Description
 test if the methods of segmaps are 
 functioning as intended
 -------------------------------------------#>
#
Class testpssegmaps : testtools {
    #
    [string]$class = 'segmaps'
    #
    testpssegmaps(){
        #
        $task = ('0', 'M21_1', $this.processloc, $this.mpath)
        $inp = segmaps $task
        #
        # Run Tests
        #
        $this.CleanupTest($inp)
        $this.SegMapsTest($inp)
        $this.testgitstatus($inp.sample)
        #
    }
    #
    [void]CleanupTest($inp){
        #
        Write-Host 'Starting Cleanup Test'
        $sor = $inp.sample.componentfolder()
        Write-Host 'Component Folder: ' $sor
        $seg_files = (get-childitem -Path $sor -Include *w_seg.tif -Recurse)
        Write-Host 'Deleting component data with segmentation data: ' $seg_files
        #
        $inp.sample.copy($sor, $this.processloc, 'W_seg.tif', 8)
        $inp.cleanup()
        $seg_files = (get-childitem -Path $sor -Include *w_seg.tif -Recurse)
        if (!($seg_files.Count -eq 0)) {
            Throw 'Error deleting component data with segmentation data'
        }
        $inp.sample.copy($this.processloc, $sor, 'w_seg.tif', 8)
        Write-Host 'Passed Cleanup Test'
        #
    }
    #
    [void]SegMapsTest($inp){
        #
        Write-Host 'Starting SegMaps Test'
        $table = $inp.sample.phenotypefolder() + '\Results\Tables'
        if (!(test-path ($table + '\*csv'))){
            Throw 'Phenotype Tables do not exist'
        }
        Write-Host 'Phenotype tables checked'
        #
        $comp = (get-childitem ($table + '\*') '*csv').Count
        $seg = (get-childitem ($inp.sample.componentfolder() + '\*') '*data_w_seg.tif').Count
        if (!($comp -eq $seg)){
            Throw 'Component data count ~= Segmentation Data count'
        }
        Write-Host 'Component data count = Segmentation data count'
        Write-Host 'Passed SegMaps Test'
        #
    }
    #
}
#
# launch test and exit if no error found
#
try {
    [testpssegmaps]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0