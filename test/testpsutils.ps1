<# -------------------------------------------
 testpsutils
 created by: Benjamin Green - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpsutils {
    #
    [string]$mpath 
    #
    testpsutils(){
        #
        $this.importmodule()
        $this.testmpath()
        $this.testapidfiles()
        $this.testsharedtools()
        #
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
    }
    #
    [void]testmpath(){
        #
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        #
        if (!(test-path $this.mpath)){
            Throw ('Cannot find mpath' + $this.mpath)
        }
        #
        Write-Host ("mpath: " + $this.mpath)
        #

    }
    #
    [void]testapidfiles(){
        #
        $apidfile = $this.mpath + '\AstropathAPIDdef.csv'
        #
        if (!(test-path $apidfile -PathType Leaf)){
            Throw ('Cannot find ap id file' + $apidfile)
        }
        #
        try {
            $apids = Get-Content $apidfile -EA Stop
        } catch {
            Throw ('Cannot open ap id file')
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
    }
    #
    [void]testsharedtools(){
        #
        try {
            $tools = sharedtools
        } catch {
            Throw 'cannot create a shared tools object'
        }

        try {
            $tools.importslideids($this.mpath)
        } Catch {
            Throw 'Cannot open apid def file'
        }
        #
    }
}
#
# launch test and exit if no error found
#
$test = [testpsutils]::new() 
exit 0
