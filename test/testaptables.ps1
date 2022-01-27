﻿<# -------------------------------------------
 testaptables
 created by: Benjamin Green - JHU
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testaptables {
    #
    [string]$mpath 
    [string]$process_loc
    #
    testaptables(){
        #
        $this.importmodule()
        $this.testmpath()
        $tools = sharedtools
        $this.testapidfiles2($tools)
        $this.testconfiginfo($tools)
        $this.correctcohortsinfo($tools)
        $this.testcohortsinfo($tools)
        #
    }
    #
    importmodule(){
        $module = $PSScriptRoot + '/../astropath'
        Import-Module $module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing_meanimage'
    }
    #
    [void]testmpath(){
        #
        if (!(test-path $this.mpath)){
            Throw ('Cannot find mpath' + $this.mpath + '. ' + $_.Exception.Message)
        }
        #
        Write-Host ("mpath: " + $this.mpath)
        #

    }
    #
    [PSCustomObject]testapidfiles(){
        #
        $apidfile = $this.mpath + '\AstropathAPIDdef.csv'
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
    testapidfiles2($tools){
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
    # the cohorts info file have to be relative to the 
    # particular branch which mean they also need to be 
    # dynamically updated
    #
    [void]correctcohortsinfo($tools){
        #
        Write-Host 'Updating cohorts info. Output below:'
        #
        $cohort_csv_template = $this.mpath + '\AstropathCohortsProgressTemplate.csv'
        $cohort_csv_file = $this.mpath + '\AstropathCohortsProgress.csv'
        $project_data = $tools.OpencsvFileConfirm($cohort_csv_template)
        $project_data[0].Dpath = $PSScriptRoot
        $project_data[1].Dpath = $PSScriptRoot + '\data'
        $project_data | Export-CSV $cohort_csv_file 
        #
        $paths_csv_template = $this.mpath + '\AstropathPathsTemplate.csv'
        $paths_csv_file = $this.mpath + '\AstropathPaths.csv'
        $paths_data = $tools.OpencsvFileConfirm($paths_csv_template)
        $paths_data[0].Dpath = $PSScriptRoot
        $paths_data[1].Dpath = $PSScriptRoot + '\data'
        $paths_data[0].FWpath = $PSScriptRoot + '\flatw'
        $paths_data[1].FWpath = $PSScriptRoot + '\data\flatw'
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
$test = [testaptables]::new() 
exit 0
