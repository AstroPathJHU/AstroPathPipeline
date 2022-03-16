  
using module .\testtools.psm1
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
Class testpsshredxml : testtools {
    #
    [string]$module = 'shredxml'
    #
    testpsshredxml() : base() {
        #
        # Setup Testing
        #
        $task = ('0', 'M21_1', $this.processloc, $this.mpath)
        $inp = shredxml $task
        #
        # Run Tests
        <#
        $this.ShredXMLTest($inp)
        $this.ReturnDataTest($inp)
        $this.CleanupTest($inp)
        $inp.sample.removedir($this.process_loc)
        $inp.sample.finish(($this.module+'test'))
        Write-Host "."
        #>
    }
    <#
    importmodule(){
        Write-Host "."
        Write-Host 'importing module ....'
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_shredxml'
    }
    #>
    [void]ShredXMLTest($inp){
        Write-Host "."
        Write-Host 'Starting Shred XML Test'
        #
        $xmlpath = $inp.processvars[1] + '\' + $inp.sample.slideid
        $userdefined = $this.process_loc, 'astropath_ws', 'shredxml', $this.slideid 
        #
        $inp.ShredXML()
        $xmlpath = $inp.processvars[1] + '\' + $inp.sample.slideid
        if (!(Test-Path $xmlpath)) {
            Throw 'Shred XML Test Failed - Path was not created'
        }
        #
        $im3s = Get-ChildItem ($inp.sample.Scanfolder() + '\MSI\*') *im3
        $im3n = ($im3s).Count + 2
        #
        # check xml files = im3s
        #
        $xmls = Get-ChildItem ($xmlpath + '\*') '*xml'
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
