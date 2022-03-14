﻿  
<# -------------------------------------------
 testpsshredxml
 created by: Andrew Jorquera
 Last Edited by: Benjamin Green
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the methods of shredxml are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsshredxml {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testpsshredxml(){
        #
        # Setup Testing
        #
        $this.importmodule()
        #
        $task = ('0', 'M21_1', $this.process_loc, $this.mpath)
        $inp = shredxml $task
        #
        # Run Tests
        #
        $this.ShredXMLTest($inp)
        $this.ReturnDataTest($inp)
        $this.CleanupTest($inp)
        $inp.sample.finish(($this.module+'test'))
        Write-Host "."
        #
    }
    #
    importmodule(){
        Write-Host "."
        Write-Host 'importing module ....'
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_shredxml'
    }
    #
    [void]ShredXMLTest($inp){
        Write-Host "."
        Write-Host 'Starting Shred XML Test'
        #
        $inp.ShredXML()
        $xmlpath = $inp.processvars[1] + '\' + $inp.sample.slideid
        if (!(Test-Path $xmlpath)) {
            Throw 'Shred XML Test Failed - Path was not created'
        }
        #
        $im3s = gci ($inp.sample.Scanfolder() + '\MSI\*') *im3
        $im3n = ($im3s).Count + 2
        #
        # check xml files = im3s
        #
        $xmls = gci ($xmlpath + '\*') '*xml'
        $files = ($xmls).Count
        if (!($im3n -eq $files)){
            Throw 'Shred XML Test Failed - Files do no match'
        }
        #
        Write-Host 'Passed Shred XML Test'
    }
    #
    [void]ReturnDataTest($inp){
        Write-Host "."
        Write-Host 'Starting Return Data Test'
        #
        $inp.returndata()
        $returnpath = $inp.sample.xmlfolder()
        #
        if (!(Test-Path $returnpath)) {
            Throw 'Return Data Test Failed - xml folder does not exist'
        }
        #
        $im3s = gci ($inp.sample.Scanfolder() + '\MSI\*') *im3
        $im3n = ($im3s).Count + 2
        #
        # check xml files = im3s
        #
        $xmls = gci ($returnpath + '\*') '*xml'
        $files = ($xmls).Count
        if (!($im3n -eq $files)){
            Throw 'Return Data Test Failed - xml files do not match'
        }
        #
        Write-Host 'Passed Return Data Test'
    }
    #
    [void]CleanupTest($inp){
        Write-Host "."
        Write-Host 'Starting Cleanup Test'
        #
        $inp.cleanup()
        if ($inp.processvars[4]) {
            if (Test-Path $inp.processloc) {
                Throw 'Cleanup Test Failed'
            }
        }
        #
        Write-Host 'Passed Cleanup Test'
    }
}
#
# launch test and exit if no error found
#
[testpsshredxml]::new() | Out-Null
exit 0

#Remove temporary processing directory
#$inp.sample.removedir($processing)
