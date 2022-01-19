﻿<# -------------------------------------------
 testpsutils
 created by: Benjamin Green - JHU
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpsutils {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testpsutils(){
        #
        $this.importmodule()
        #
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
    }
    #
    [void]testloggerconstruction(){
        #
        try {
            $log = logger
        } catch {
            Throw 'logger contruction with [0] input(s) failed'
        }
        #
        try {
            $log = logger $this.mpath 'shredxml'
        } catch {
            Throw 'logger contruction with [2] input(s) failed'
        }
        #
        try {
            $log = logger $this.mpath 'shredxml' 'M21_1'
        } catch {
            Throw 'logger contruction with [3] input(s) failed'
        }
        #
        try {
            $log = logger $this.mpath 'shredxml' 'M21_1' '0'
        } catch {
            Throw 'logger contruction with [4] input(s) failed'
        }
        #
    }
}
#
# launch test and exit if no error found
#
$test = [testpsutils]::new() 
exit 0
