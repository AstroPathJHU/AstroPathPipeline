<# -------------------------------------------
 testpslogger
 created by: Benjamin Green - JHU
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpslogger {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testpslogger(){
        #
        $this.importmodule()
        $this.testloggerconstruction()
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
            Throw ('[logger] construction with [0] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            $log = logger $this.mpath 'shredxml'
        } catch {
            Throw ('[logger] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            $log = logger $this.mpath 'shredxml' 'M21_1'
        } catch {
            Throw ('[logger] construction with [3] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            $log = logger $this.mpath 'shredxml' '08' '0'
        } catch {
            Throw ('[logger] construction with [4] input(s) failed. ' + $_.Exception.Message)
        }
        #
    }
}
#
# launch test and exit if no error found
#
$test = [testpslogger]::new() 
exit 0
