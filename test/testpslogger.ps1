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
        $log = logger $this.mpath 'shredxml' 'M21_1'
        $log.basepath = $PSScriptRoot + '\data'
        $log.defpaths()
        Write-Host $log.project_data
        #
        $this.testwritestartmessage($log)
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
        Write-Host '[logger] construction tests started'
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
            $log = logger $this.mpath 'shredxml' '01' '0'
        } catch {
            Throw ('[logger] construction with [4] input(s) failed. ' + $_.Exception.Message)
        }
        #
        Write-Host '[logger] construction tests finished'
        #
    }
    #
    [void]testwritestartmessage($log){
        #
        Write-Host 'write to log tests started'
        #
        Write-Host 'write to main log'
        #
        $log.level = 4
        #
        $log.Start('shredxml-test')
        #
        Write-Host 'write to console log'
        #
        $log.level = 8
        #
        $log.Start('shredxml-test')
        #
        $log.level = 12
        #
        Write-Host 'write to main log and console'
        #
        $log.Start('shredxml-test')
        #
        $log.level = 2
        #
        Write-Host 'write to slide log'
        #
        $log.Start('shredxml-test')
        #
        Write-Host 'write to log tests finished'
        #
    }
    #
    [void]correctlogger($log){
        #
        $log.basepath = $PSScriptRoot + '\data'
        $log.defpaths()
        Write-Host $log.project_data
        #
    }
}
#
# launch test and exit if no error found
#
$test = [testpslogger]::new() 
exit 0
