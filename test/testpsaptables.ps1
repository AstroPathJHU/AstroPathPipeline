using module .\testtools.psm1
<# -------------------------------------------
 testaptables
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpsaptables : testtools {
    #
    [string]$class = 'aptabletools'
    #
    testpsaptables() : base(){
        #
        $this.testmpath()
        $tools = sharedtools
        $this.testapidfiles() | Out-Null
        $this.testapidfiles($tools)
        $this.testconfiginfo($tools)
        $this.testcorrectionfile($tools, $true)
        $this.testcorrectioninfo($tools)
        $this.correctcohortsinfo($tools)
       # $this.testcohortsinfo($tools)
        $this.testgitstatus($tools)        
        Write-Host '.'
        #
    }
    #
    [void]testmpath(){
        #
        Write-Host '.'
        Write-Host 'test mpath'
        if (!(test-path $this.mpath)){
            Throw ('Cannot find mpath' + $this.mpath + '. ' + $_.Exception.Message)
        }
        #
        Write-Host ("mpath: " + $this.mpath)
        Write-Host (get-childitem $this.mpath | Format-Table | Out-String)
        #

    }
    #
    [PSCustomObject]testapidfiles(){
        #
        Write-Host '.'
        Write-Host 'Testing manual apid import. Output below:'
        $apidfile = $this.mpath + '\AstroPathAPIDdef.csv'
        #
        if (!(test-path $apidfile -PathType Leaf)){
            Throw ('Cannot find ap id file' + $apidfile + '. ' + $_.Exception.Message)
        }
        #
        try {
            $apids = Import-CSV $apidfile -EA Stop
        } catch {
            Throw ('Cannot open ap id file. ' + $_.Exception.Message)
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
        return $apids
        #
    }
    #
    testapidfiles($tools){
        Write-Host '.'
        Write-Host 'Testing import slideids method. Output below:'
        #
        try {
            $internal_apids = $tools.importslideids($this.mpath)
        } Catch {
            Throw ('Cannot open apid def file. ' + $_.Exception.Message)
        }
        #
        write-host " " ($internal_apids | 
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
    [void]testconfiginfo($tools){
        #
        Write-Host '.'
        Write-Host 'Testing config info method. Output below:'
        #
        try {
            $internal_apids = $tools.ImportConfigInfo($this.mpath)
        } Catch {
            Throw ('Cannot open config file. ' + $_.Exception.Message)
        }
        #
        write-host " " $internal_apids | Format-Table
        #
    }
    #
    [void]testcohortsinfo($tools){
        #
        Write-Host '.'
        Write-Host 'Testing Cohorts info method. Output below:'
        #
        try {
            $internal_apids = $tools.ImportCohortsInfo($this.mpath)
        } Catch {
            Throw ('Cannot open config file. ' + $_.Exception.Message)
        }
        #
        write-host " " ($internal_apids | Out-String)
        #
    }
    #
    [void]testcorrectioninfo($tools){
        #
        Write-Host '.'
        Write-Host 'Testing import correction models method. Output below:'
        #
        $ids = $tools.ImportCorrectionModels($this.mpath)
        #
        Write-Host '    test models csv exists:' (test-path ($this.mpath + '\AstroPathCorrectionModels.csv'))
        #
        if (!$ids){
            Throw 'correction models is empty!!'
        }
        Write-Host '    correction models file:'
        Write-Host '    ' ($ids | Format-Table | Out-String)
    }
    #
    # the cohorts info file have to be relative to the 
    # particular branch which mean they also need to be 
    # dynamically updated
    #
    [void]correctcohortsinfo($tools){
        #
        Write-Host '.'
        Write-Host 'Updating cohorts info. Output below:'
        #
        $cohort_csv_file =  $tools.cohorts_fullfile($this.mpath) 
        $cohort_csv_template = $cohort_csv_file -replace $tools.apfile_constant, $this.apfile_temp_constant
        $project_data = $tools.OpencsvFileConfirm($cohort_csv_template)
        $p = $this.uncpath($PSScriptRoot)
        $project_data[0].Dpath = $p 
        $project_data[1].Dpath = $p  + '\data'
        $project_data | Export-CSV $cohort_csv_file 
        #
        $paths_csv_file =  $tools.paths_fullfile($this.mpath) 
        $paths_csv_template = $paths_csv_file -replace $tools.apfile_constant, $this.apfile_temp_constant
        $paths_data = $tools.OpencsvFileConfirm($paths_csv_template)
        $paths_data[0].Dpath = $p 
        $paths_data[1].Dpath = $p + '\data'
        $paths_data[0].FWpath = ($p  + '\flatw') -replace '\\\\', ''
        $paths_data[1].FWpath = ($p  + '\data\flatw') -replace '\\\\', ''
        $paths_data | Export-CSV $paths_csv_file
        #
        $internal_apids = $tools.ImportCohortsInfo($this.mpath)
        write-host " " ($internal_apids | Out-String)
        #
    }
    #
}
#
# launch test and exit if no error found
#
try {
    [testpsaptables]::new() | Out-Null
} catch {
    Throw $_.Exception
}
exit 0

