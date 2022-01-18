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
    [string]$mpath 
    #
    testpsimport(){
        #
        $this.testimport()
        $this.testmpath()
        $this.testapidfiles()
        #
    }
    #
    [void]testimport(){
      #
      $module = $PSScriptRoot + '/../astropath'
      Write-Host 'checking: ' $module
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
    #
    [void]testmpath(){
        #
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        #
        if (!(test-path $this.mpath)){
            Throw ('Cannot find mpath' + $this.mpath)
        }
        #
        Write-Host $this.mpath
        #

    }
    #
    [void]testapidfiles(){
        #
        $apidfile = $this.mpath + '\AstropathAPIDdef.csv'
        #
        if (!($apidfile)){
            Throw ('Cannot find ap id file' + $apidfile)
        }
        #
        try {
            $apids = Get-Content $apidfile
        } catch {
            Throw ('Cannot open ap id file')
        }
        #
        write-host " " ($apids | 
            Format-Table  @{Name="SlideID";Expression = { $_.SlideID }; Alignment="center" },
                            @{Name="SampleName";Expression = { $_.SampleName }; Alignment="center" },
                            @{Name="Project";Expression = { $_.Project }; Alignment="center" },
                            @{Name="Cohort";Expression = { $_.Cohort }; Alignment="center" },
                            @{Name="Scan";Expression = { $_.Scan }; Alignment="center" },
                            @{Name="BatchID";Expression = { $_.BatchID }; Alignment="center" }, 
                            @{Name="isGood";Expression = { $_.isGood }; Alignment="center" } |
            Out-String).Trim() -ForegroundColor Yellow
        #
    }
    #
}
#
# launch test and exit if no error found
#
$test = [testpsimport]::new() 
exit 0
