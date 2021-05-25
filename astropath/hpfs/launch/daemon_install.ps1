<# ---------------------------------------------------------
daemon_install.ps1

script to download necessary Server Demon packages

Usage:
      daemon_install.ps1
            
---------------------------------------------------------#>
function daemon_install{
      Write-Host "Installing Daemon Packages..."
      #
      # add the current version of python to the path
      #
      $env:Path += ";C:\Program Files (x86)\Python38-32";
      $env:PATHEXT += ";.py"; 
      #
      $code_path = "$PSScriptRoot\..\TransferDaemon\setup.py"
      #
      python $code_path install
      #
      Write-Host "Packages Finished Installing"
}
#
# run the function
#
daemon_install