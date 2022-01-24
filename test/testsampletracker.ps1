 <# -------------------------------------------
 testsampletracker
 created by: Benjamin Green - JHU
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testsampletracker {
    #
    [string]$mpath 
    [string]$module 
    [string]$process_loc
    #
    testsampletracker(){
        #
        $this.importmodule()
        $this.testsampletrackerconstructors()
        $sampletracker = sampletracker -mpath $this.mpath -slideid 'M21_1'
        $this.testmodules($sampletracker)
        $this.teststatus($sampletracker)
        #
    }
    #
    importmodule(){
        $this.module = $PSScriptRoot + '/../astropath'
        Import-Module $this.module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
    }
    #
    [void]testsampletrackerconstructors(){
        #
        Write-Host 'test [sampletracker] constructors started'
        #
        try{
            $sampletracker = sampletracker -mpath $this.mpath -slideid 'M21_1'
            $sampletracker.removewatchers()
        } catch {
            Throw ('[sampletracker] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #
        Write-Host 'test [sampletracker] constructors finished'
        #                
    }
    #
    [void]testmodules($sampletracker){
        #
        try {
            $sampletracker.getmodulenames()
        } catch {
            Throw ('[sampletracker].getmodulenames() failed: ' + $_.Exception.Message)
        }
        #
        Write-Host $sampletracker.modules 
        #
        $cmodules = @('batchflatfield','batchmicomp','imagecorrection','meanimage','mergeloop',`
            'segmaps','shredxml','transfer','vminform','warpoctets')
        $out = Compare-Object -ReferenceObject $sampletracker.modules  -DifferenceObject $cmodules
        if ($out){
            Throw ('module lists in [sampletracker] does not match, this may indicate new modules or a typo:' + $out)
        }
        #
    }
    #
    [void]teststatus($sampletracker){
        #
        try {
            $sampletracker.defmodulestatus()
        } catch {
            Throw ('[sampletracker].defmodulestatus() failed: ' + $_.Exception.Message)
        }
        #
        Write-Host $sampletracker.modulestatus
        #
    }
    #
    [void]testupdate(){

    }
    #
}
#
# launch test and exit if no error found
#
$test = [testsampletracker]::new() 
exit 0
