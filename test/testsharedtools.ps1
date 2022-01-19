 <# -------------------------------------------
 testsharedtools
 created by: Benjamin Green - JHU
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testsharedtools {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testsharedtools(){
        #
        $this.importmodule()
        $this.testconstructor()
        $tools = sharedtools
        $this.testcheckgitrepo($tools)
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
    testconstructor(){
        #
        try {
            $tools = sharedtools
        } catch {
            Throw ('cannot create a [sharedtools] object. ' + $_.Exception.Message)
        }
        #
        Write-Host '[sharedtools] object created'
        #
    }
    #
    [void]testcheckgitrepo($tools){
        #
        Write-Host 'root: ' $tools.defRoot()
        Write-Host 'OS: ' $tools.defOS()
        Write-Host 'py package path: ' $tools.pypackagepath()
        Write-Host 'Git installed: ' $tools.checkgitinstalled()
        git -C $tools.pypackagepath() status
        Write-Host 'Git repo: ' $tools.checkgitrepo()
        #
    }
}
#
# launch test and exit if no error found
#
$test = [testsharedtools]::new() 
exit 0
