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
    testpsimport(){
      #
      $module = $PSScriptRoot + '/../astropath'
      Write-Host 'checking: ' $module
      $files = gci $module
      Write-Host $files
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
}
#
# launch test and exit if no error found
#
$test = [testpsimport]::new() 
exit 0
