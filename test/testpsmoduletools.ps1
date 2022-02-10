<# testpsmoduletools
 testpslogger
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpsmoduletools {
    #
    [string]$mpath 
    [string]$process_loc
    [string]$basepath = $PSScriptRoot + '\data'
    [string]$module = 'shredxml'
    [string]$slideid = 'M21_1'
    [string]$project = '0'
    [string]$apmodule = $PSScriptRoot + '\..\astropath'
    #
    testpsmoduletools(){
        #
        Write-Host '---------------------test ps [moduletools]---------------------'
        $this.importmodule()
        $this.testmodulecontruction()
        #
        $task = ($this.project, $this.slideid, $this.process_loc, $this.mpath)
        $inp = meanimage $task
        #
        $this.TestPaths($inp)
        Write-Host '.'
        #
    }
    #
    importmodule(){
        Import-Module $this.apmodule -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing'
    }
    #
    [void]testmodulecontruction(){
        #
        Write-Host '.'
        Write-Host 'building a shredxml module object'
        try {
            $task = ($this.project, $this.slideid, $this.process_loc, $this.mpath)
            shredxml $task | Out-Null
        } catch {
            Throw 'module could not be constructed'
        }
        #
    }
    #
    [void]TestPaths($inp){
        #
        Write-Host '.'
        Write-Host 'Starting Paths Testing'
        #
        $testloc = $this.process_loc + '\astropath_ws\meanimage\' + $this.slideid
        #
        if (!([regex]::Escape($inp.processvars[0]) -contains [regex]::Escape($testloc))){
            Throw ('processvars[0] not correct: ' + $inp.processvars[0] + '~=' + $testloc)
        }
        #
        if (!([regex]::Escape($inp.processvars[1]) -contains [regex]::Escape(($testloc + '\flatw')))){
            Throw ('processvars[1] not correct: ' + $inp.processvars[1] + '~=' + $testloc + '\flatw')
        }
        #
        if (!([regex]::Escape($inp.processvars[2]) -contains [regex]::Escape(($testloc + '\' + $this.slideid + '\im3\flatw')))){
            Throw ('processvars[2] not correct: ' + $inp.processvars[2] + '~=' + $testloc + '\' + $this.slideid + '\im3\flatw')
        }
        #
        if (!([regex]::Escape($inp.processvars[3]) -contains [regex]::Escape(($testloc + '\flatfield\flatfield_BatchID_08.bin')))){
            Write-Host 'batch flatfield file:' $inp.sample.batchflatfield()
            Throw ('processvars[3] not correct: ' + $inp.processvars[3] + '~=' + $testloc + '\flatfield\flatfield_BatchID_08.bin')
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
[testpsmoduletools]::new() | Out-Null
exit 0
