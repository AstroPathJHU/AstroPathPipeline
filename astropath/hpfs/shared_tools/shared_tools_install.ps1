<# ---------------------------------------------------------
shared_tools_install.ps1

script to install the shared_tools module

Usage:
      shared_tools_install.ps1
            
---------------------------------------------------------#>
function shared_tools_install{
      Write-Host "Installing shared_tools..."
      #
      # add the current version of python to the path
      #
      $env:Path += ";C:\Program Files (x86)\Python38-32";
      $env:PATHEXT += ";.py"; 
      #
      $code_path = "$PSScriptRoot\..\shared_tools\setup.py"
      #
      python $code_path install
      #
      Write-Host "shared_tools Finished Installing"
}
#
# run the function
#
shared_tools_install