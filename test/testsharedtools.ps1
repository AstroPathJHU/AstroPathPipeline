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
        $this.testconstructor()
        $tools = sharedtools
        #
    }
    #
    [void]testconstructor(){
        #
        try {
            $tools = sharedtools
        } catch {
            Throw 'cannot create a shared tools object'
        }
        #
        Write-Host '[sharedtools] object created'
        #
    }
    #
    [void]testcheckgitrepo($tools){
        #
        Write-Host 'py package path: ' $this.pypackagepath()
        Write-Host 'Git installed: ' $tools.checkgitinstalled()
        Write-Host 'Git repo: ' $tools.checkgitrepo()
        #
    }
}
#
# launch test and exit if no error found
#
$test = [testsharedtools]::new() 
exit 0
