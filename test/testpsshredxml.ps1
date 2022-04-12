using module .\testtools.psm1
<# -------------------------------------------
 testpsshredxml
 Andrew Jorquera, Benjamin Green
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
    [string]$class = 'shredxml'
    #
    testpsshredxml() : base() {
        #
        # Setup Testing
        #
        $task = ($this.project, $this.slideid, $this.processloc, $this.mpath)
        $inp = shredxml $task
        #
        # Run Tests
        #
        $this.testprocessroot($inp)
        $this.ReturnDataTest($inp)
        $this.CleanupTest($inp)
        $inp.sample.finish(($this.module+'test'))
        $this.testgitstatus($inp.sample)  
        Write-Host "."
        #
    }
    #
    [void]testprocessroot($inp){
        #
        Write-Host "."
        Write-Host 'test process root started'
        #
        $xmlpath = $inp.processvars[1]
        $userdefined = $this.processloc, 'astropath_ws', 'shredxml', $this.slideid -join '\'
        #
        Write-Host '    user defined:' $userdefined
        Write-Host '    [shredxml] defined:' $xmlpath
        #
        if (!(Test-Path -LiteralPath $xmlpath)) {
            Throw 'Shred XML Test Failed - XML Path was not created'
        }
        #
        if ($xmlpath -notmatch [regex]::escape($userdefined)){
            Throw 'process dir not defined correctly'
        }
        #
        Write-Host 'test process root finished'
        #
    }
    #
    [void]ReturnDataTest($inp){
        #
        Write-Host "."
        Write-Host 'Return Data Test started'
        #
        Write-Host '    copy old results to a safe location'
        $sor = $this.basepath, $this.slideid, 'im3\xml' -join '\'
        $des = $this.processloc, $this.slideid, 'im3\xml' -join '\'
        #
        Write-Host '    source:' $sor
        Write-Host '    destination:' $des
        $inp.sample.copy($sor, $des, '*')       
        #
        Write-Host '    copy old results to processing directory'
        $userdefined = $this.processloc, 'astropath_ws',
            'shredxml', $this.slideid, $this.slideid -join '\'
        $inp.sample.copy($sor, $userdefined, '*')  
        #
        Write-Host '    run return data'
        $inp.returndata()
        #
        $this.comparepaths($des, $sor, $inp)
        #
        Write-Host 'Return Data Test finished'
        #
    }
    #
    [void]CleanupTest($inp){
        #
        Write-Host "."
        Write-Host 'Cleanup Test started'
        #
        Write-Host '    running clean up'
        Write-Host '    expected to remove:' $inp.processloc
        #
        $inp.cleanup()
        #
        if (Test-Path $inp.processloc) {
            Throw 'Cleanup Test Failed dir still exists'
        }
        #
        Write-Host '    cleaning up after test'
        Write-Host '    expected to remove:' $this.processloc
        $inp.sample.removedir($this.processloc)
        #
        if (Test-Path $this.processloc) {
            Throw 'Cleanup Test Failed dir still exists'
        }
        #
        Write-Host 'Cleanup Test finished'
        #
    }
}
#
# launch test and exit if no error found
#
try {
    [testpsshredxml]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0
