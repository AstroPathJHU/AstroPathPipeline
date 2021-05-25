<# ---------------------------------------------------------
launch_transferdeamon.ps1

script to run the Server Demon

Usage:
      launch_transferdaemon.ps1 -csv -email [-test]
      args 
            -csv
            -email
      Optional args
            [-d]
            
---------------------------------------------------------#>
#
# d indicates the usage of debug mode
#
param([Parameter(Position=0)][string] $csv = '',
      [Parameter(Position=1)][string] $email = '',
      [Parameter()][switch]$d
      )
#
# check input
#
if (
      !($PSBoundParameters.ContainsKey('csv')) -OR
      !($PSBoundParameters.ContainsKey('email'))
   ) {
      Write-Host "Usage: launch_transferdeamon mpath email [-test]"; return
      }
function transferdeamon{
      #
      param([Parameter(Position=0)][string] $csv = '',
            [Parameter(Position=1)][string] $email = '',
            [Parameter()][switch]$d
            )
      #
      # add the current version of python to the path
      #
      $env:Path += ";C:\Program Files (x86)\Python38-32";
      $env:PATHEXT += ";.py"; 
      #
      $code_path = "$PSScriptRoot\..\TransferDaemon\Daemon.py"
      #
      # run code in test mode
      #
      if ($d){
            Write-Host "StartSeverDemon", $csv, $email, -d, -nonewline
            python $code_path $csv $email -d
      } else {
            Write-Host "StartSeverDemon", $csv, $email, -nonewline
            python $code_path $csv $email
      }
}
#
# run the function
#
transferdeamon $csv $email -d:$d
