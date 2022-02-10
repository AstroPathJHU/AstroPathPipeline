<# -------------------------------------------
 testsampledb
 created by: Benjamin Green - JHU
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test the sampledb is working as expected
 -------------------------------------------#>
#
 Class testsampledb {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testsampledb(){
        #
        $this.importmodule()

    
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing'
    }
    #
    [void]testsampledbconstructors(){
        #
        Write-Host 'test [sampledb] constructors started'
        #
        try{
            $sampledb = sampledb -mpath $this.mpath
        } catch {
            Throw ('[sampledb] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try{
            $sampledb = sampledb -mpath $this.mpath -projects '00'
        } catch {
            Throw ('[sampledb] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #
        Write-Host 'test [sampledb] constructors finished'
        #                
    }
    #
    [void]
}
#
# launch test and exit if no error found
#
$test = [testpslogger]::new() 
exit 0
