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
$test = [testpsimport]::new($modulepath) 
exit 0
