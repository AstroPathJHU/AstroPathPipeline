  
<# -------------------------------------------
 testpsshredxml
 created by: Andrew Jorquera
 Last Edited by: Benjamin Green
 Last Edit: 1/10/2022
 --------------------------------------------
 Description
 test if the methods of shredxml are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsshredxml {
    #
    testpsshredxml(){
        #
        # Setup Testing
        #
        $module = '\\bki08\e$\working_code\dev\AstroPathPipelinePrivate\astropath'
        Import-Module $module -EA SilentlyContinue
        #
        $processing = '\\Bki08\h\testing'
        $mpath = '\\Bki08\h\testing\astropath_processing'
        $task = ('1', 'M21_1', $processing, $mpath)
        $inp = shredxml $task
        #
        # Run Tests
        #
        $this.TestPaths($inp)
        $this.ShredXMLTest($inp)
        $this.ReturnDataTest($inp)
        $this.CleanupTest($inp)
        #
    }
    #
    [void]TestPaths($inp){
        Write-Host 'Starting Paths Testing'
        #
        if (!([regex]::Escape($inp.processvars[0]) -contains [regex]::Escape('\\bki08\h$\testing'))){
            Throw 'processvars[0] not correct'
        }
        #
        if (!([regex]::Escape($inp.processvars[1]) -contains [regex]::Escape('\\Bki08\h\testing\astropath_ws\shredxml\M21_1\flatw'))){
            Throw 'processvars[1] not correct'
        }
        #
        if (!([regex]::Escape($inp.processvars[2]) -contains [regex]::Escape('\\Bki08\h\testing\astropath_ws\shredxml\M21_1\M21_1\im3\flatw'))){
            Throw 'processvars[2] not correct'
        }
        #
        if (!([regex]::Escape($inp.processvars[3]) -contains [regex]::Escape('\\Bki08\h\testing\astropath_ws\shredxml\M21_1\flatfield\flatfield_BatchID_99.bin'))){
            Throw 'processvars[3] not correct'
        }
        #
        if (!([regex]::Escape($inp.processloc) -contains [regex]::Escape('\\bki08\h\testing\astropath_ws\shredxml\M21_1'))){
            Throw 'processloc not correct'
        }
        #
        if (!(test-path $inp.processloc)){
            Throw 'processloc does not exist'
        }
        #
        Write-Host 'Passed Paths Testing'
        #
    }
    #
    [void]ShredXMLTest($inp){
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
$test = [testpsmeanimage]::new()
exit 0

#Remove temporary processing directory
#$inp.sample.removedir($processing)
