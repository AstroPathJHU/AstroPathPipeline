<# -------------------------------------------
 testaptables
 created by: Benjamin Green - JHU
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testaptables {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testaptables(){
        #
        $this.importmodule()
        $this.testmpath()
        $tools = sharedtools
        $this.testapidfiles2($tools)
        $this.testconfiginfo($tools)
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
    [void]testmpath(){
        #
        if (!(test-path $this.mpath)){
            Throw ('Cannot find mpath' + $this.mpath + '. ' + $_.Exception.Message)
        }
        #
        Write-Host ("mpath: " + $this.mpath)
        #

    }
    #
    [PSCustomObject]testapidfiles(){
        #
        $apidfile = $this.mpath + '\AstropathAPIDdef.csv'
        #
        if (!(test-path $apidfile -PathType Leaf)){
            Throw ('Cannot find ap id file' + $apidfile + '. ' + $_.Exception.Message)
        }
        #
        try {
            $apids = Import-CSV $apidfile -EA Stop
        } catch {
            Throw ('Cannot open ap id file. ' + $_.Exception.Message)
        }
        #
        write-host " " ($apids | 
            Format-Table  @{Name="SlideID";Expression = { $_.SlideID }; Alignment="center" },
                            @{Name="SampleName";Expression = { $_.SampleName }; Alignment="center" },
                            @{Name="Project";Expression = { $_.Project }; Alignment="center" },
                            @{Name="Cohort";Expression = { $_.Cohort }; Alignment="center" },
                            @{Name="Scan";Expression = { $_.Scan }; Alignment="center" },
                            @{Name="BatchID";Expression = { $_.BatchID }; Alignment="center" }, 
                            @{Name="isGood";Expression = { $_.isGood }; Alignment="center" } |
            Out-String).Trim() -ForegroundColor Yellow
        #
        return $apids
        #
    }
    #
    testapidfiles2($tools){
        Write-Host 'Testing import slideids method. Output below:'
        #
        try {
            $internal_apids = $tools.importslideids($this.mpath)
        } Catch {
            Throw ('Cannot open apid def file. ' + $_.Exception.Message)
        }
        #
        write-host " " ($internal_apids | 
        Format-Table  @{Name="SlideID";Expression = { $_.SlideID }; Alignment="center" },
                        @{Name="SampleName";Expression = { $_.SampleName }; Alignment="center" },
                        @{Name="Project";Expression = { $_.Project }; Alignment="center" },
                        @{Name="Cohort";Expression = { $_.Cohort }; Alignment="center" },
                        @{Name="Scan";Expression = { $_.Scan }; Alignment="center" },
                        @{Name="BatchID";Expression = { $_.BatchID }; Alignment="center" }, 
                        @{Name="isGood";Expression = { $_.isGood }; Alignment="center" } |
        Out-String).Trim() -ForegroundColor Yellow
        #
    }
    #
    [void]testconfiginfo($tools){
        #
        Write-Host 'Testing config info method. Output below:'
        #
        try {
            $internal_apids = $tools.ImportConfigInfo($this.mpath)
        } Catch {
            Throw ('Cannot open config file. ' + $_.Exception.Message)
        }
        #
        write-host " " $internal_apids 
        #
    }
    #
}
#
# launch test and exit if no error found
#
$test = [testaptables]::new() 
exit 0
