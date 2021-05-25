<# ---------------------------------------------------------
astroidgen_install.ps1

script to download necessary Astroid Generator packages

Usage:
      astroidgen_install.ps1
            
---------------------------------------------------------#>
function astroidgen_install{
      Write-Host "Installing ASTgen Packages..."
      #
      # add the current version of python to the path
      #
      $env:Path += ";C:\Program Files (x86)\Python38-32";
      $env:PATHEXT += ";.py"; 
      #
      $code_path = "$PSScriptRoot\..\AstroidGen\setup.py"
      #
      python $code_path install
      #
      Write-Host "Packages Finished Installing"
}
#
# run the function
#
astroidgen_install