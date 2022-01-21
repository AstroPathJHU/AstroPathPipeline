<# -------------------------------------------
 testpsimport
 created by: Benjamin Green - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpsimport {
    #
    [string]$mpath 
    #
    testpsimport(){
      #
      $module = $PSScriptRoot + '/../astropath'
      Write-Host 'checking: ' $module
      #
      # check for the module
      #
      $modules = Get-Module -ListAvailable -Name $module 
      if ($modules) {
            Write-Host "Module exists"
            Write-Host $modules
      } else {
          Throw "Module does not exist"
      }
      #
      # confirm installation
      #
      Import-Module $module -EA SilentlyContinue
      if($error){
          Throw 'Module could not be imported'
      } 
      #
    }
    #
    testmpath(){
        #
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        #
        if (!(test-path $this.mpath)){
            Throw ('Cannot find mpath' + $this.mpath)
        }
        #
        Write-Host $this.mpath
        #

    }
    #
    testapidfiles(){
        #
        $apidfile = $this.mpath + '\AstroPathAPIDdef.csv'
        #
        if (!($apidfile)){
            Throw ('Cannot find ap id file' + $apidfile)
        }
        #
    }
}
#
# launch test and exit if no error found
#
$test = [testpsimport]::new() 
exit 0
