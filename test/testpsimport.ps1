<# -------------------------------------------
 testpsimport
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpsimport {
    #
    [string]$mpath 
    [string]$process_loc
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    #
    testpsimport(){
        #
        Write-Host '---------------------test ps astropath module import---------------------'
        $this.testimport()
        Write-Host '.'
        #
    }
    #
    [void]testimport(){
      #
      Write-Host '.'
      Write-Host 'checking: ' $this.apmodule
      #
      # check for the module
      #
      $modules = Get-Module -ListAvailable -Name $this.apmodule
      if ($modules) {
            Write-Host "Module exists"
            Write-Host $modules
      } else {
          Throw "Module does not exist"
      }
      #
      # confirm installation
      #
      Import-Module $this.apmodule -EA SilentlyContinue
      if($error){
          Throw 'Module could not be imported'
      } 
      #
      Write-Host 'module imported successfully'
    }
    #
}
#
# launch test and exit if no error found
#
[testpsimport]::new() | Out-Null
exit $LASTEXITCODE
