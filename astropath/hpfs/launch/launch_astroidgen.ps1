<# ---------------------------------------------------------
launch_astroidgen.ps1

script to run the AstroID Generator

Usage:
      launch_astroidgen.ps1 -csv
      args 
            -csv
            
---------------------------------------------------------#>
param([Parameter(Position=0)][string] $csv = '')
#
# check input
#
if (
      !($PSBoundParameters.ContainsKey('csv'))
   ) {
      Write-Host "Usage: launch_transferdeamon mpath"; return
      }
function astroidgen{
      #
      param([Parameter(Position=0)][string] $csv = '')
      #
      # add the current version of python to the path
      #
      $env:Path += ";C:\Program Files (x86)\Python38-32";
      $env:PATHEXT += ";.py"; 
      #
      $code_path = "$PSScriptRoot\..\astroidgen\ASTgen.py"
      #
      python $code_path $csv
}
#
# run the function
#
launch_astroidgen $csv


