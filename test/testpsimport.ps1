<# -------------------------------------------
 testpsimport
 created by: Benjamin Green - JHU
 Last Edit: 10.13.2020
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
 param ([Parameter(Position=0)][string] $modulepath = '')
#
# check input parameters
#
if (!($PSBoundParameters.ContainsKey('modulepath'))) {
    Throw "Usage: testpsimport -modulepath"
}
#
 Class testpsimport {
    #
    testpsimport($modulepath){
      #
      $module = $modulepath + '/../astropath'
      Write-Host 'checking: ' $module
      #
      if (Get-Module -ListAvailable -Name $module) {
            Write-Host "Module exists"
            Write-Host Get-Module -ListAvailable -Name $module
      } else {
          Throw "Module does not exist"
      }
      #
      try {
           Import-Module $module -EA SilentlyContinue
      } catch {
          Throw "Module import failed"
      }
      #
    }
}
#
$test = [testpsimport]::new($modulepath) 
exit 0
