<# testpsmoduletools
 testpslogger
 created by: Benjamin Green - JHU
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpsmoduletools {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testpsmoduletools(){
        #
        $this.importmodule()
        $this.testmodulecontruction()
        #
        $task = ('0', 'M21_1', $this.process_loc, $this.mpath)
        $inp = meanimage $task
        #
        $this.TestPaths($inp)
        #
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '\..\astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins'
    }
    #
    [void]testmodulecontruction(){
        #
        Write-Host 'building a shredxml module object'
        try {
            $task = ('0', 'M21_1', $this.process_loc, $this.mpath)
            $inp = shredxml $task
        } catch {
            Throw 'module could not be constructed'
        }
        #
    }
    #
    [void]TestPaths($inp){
        Write-Host 'Starting Paths Testing'
        #
        $testloc = $this.process_loc + '\astropath_ws\meanimage'
        #
        if (!([regex]::Escape($inp.processvars[0]) -contains [regex]::Escape($testloc))){
            Throw ('processvars[0] not correct: ' + $inp.processvars[0] + '~=' + $testloc)
        }
        #
        if (!([regex]::Escape($inp.processvars[1]) -contains [regex]::Escape(($testloc + '\M21_1\flatw')))){
            Throw ('processvars[1] not correct: ' + $inp.processvars[0] + '~=' + $testloc + '\M21_1\flatw')
        }
        #
        if (!([regex]::Escape($inp.processvars[2]) -contains [regex]::Escape(($testloc + '\M21_1\M21_1\im3\flatw')))){
            Throw ('processvars[2] not correct: ' + $inp.processvars[0] + '~=' + $testloc + '\M21_1\M21_1\im3\flatw')
        }
        #
        if (!([regex]::Escape($inp.processvars[3]) -contains [regex]::Escape(($testloc + '\M21_1\flatfield\flatfield_BatchID_8.bin')))){
            Throw ('processvars[3] not correct: ' + $inp.processvars[0] + '~=' + $testloc + '\M21_1\flatfield\flatfield_BatchID_8.bin')
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
}
#
# launch test and exit if no error found
#
$test = [testpsmoduletools]::new() 
exit 0
