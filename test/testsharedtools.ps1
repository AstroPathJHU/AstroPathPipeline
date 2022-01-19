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
        Write-Host 'is Windows: ' $tools.isWindows()
        Write-Host 'py package path: ' $tools.pypackagepath() 
        Write-Host 'Git installed: ' $tools.checkgitinstalled()
        Write-Host 'Git repo: ' $tools.checkgitrepo()
        Write-Host 'Git version: ' $tools.getgitversion()
        Write-Host 'Git status: ' $tools.checkgitstatus() 
        # Write-Host 'Git full version: ' $tools.getfullversion()
        #
        Set-Content (($tools.pypackagepath() + '/file.csv') -replace ('/', '\')) $tools.pypackagepath()
        #
    }
}
#
# launch test and exit if no error found
#
$test = [testsharedtools]::new() 
exit 0
