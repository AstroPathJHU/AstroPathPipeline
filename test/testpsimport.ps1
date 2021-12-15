<# -------------------------------------------
 testpsimport
 created by: Benjamin Green - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 copy files fast with different methods
 -------------------------------------------#>
 Class testpsimport {
    #
    testpsimport(){
      $module = $PSScriptRoot + '../astropath'
      Write-Host 'checking: ' $module
      if (Get-Module -ListAvailable -Name $module) {
            Write-Host "Module exists"
      } 
      else {
          Throw "Module does not exist"
      }
    }
}
#
$test = [testpsimport]::new() 
