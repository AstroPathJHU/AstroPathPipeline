<# -------------------------------------------
 testpspyupdate
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpspyupdate {
    #
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    #
    testpspyupdate(){
        #
        Write-Host '---------------------update ps [py envir]---------------------'
        $this.importmodule()
        $tools = sharedtools
        $tools.UpgradepyEnvir()
        Write-Host '.'
        #
    }
    #
    importmodule(){
        Import-Module $this.apmodule
    }
    #
}
#
# launch test and exit if no error found
#
[testpspyupdate]::new() | Out-Null
exit 0
