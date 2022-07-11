<# -------------------------------------------
 aptabletools
 created by: Benjamin Green, Andrew Jorquera 
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used assess, open and process the 
 astropath tables in the astropath processing
 so called <mpath> directory. 

 Note that:
 logic to replace where-object with process block is from:
 https://powershell.one/tricks/performance/pipeline
 logic may be used elsewhere in astropath module.
 dramatic preformance improvement
 -------------------------------------------#>
class aptabletools : fileutils {
    #
    [PSCustomObject]$full_project_dat
    [PSCustomObject]$dependency_data
    [PSCustomObject]$micomp_data
    [PSCustomObject]$corrmodels_data
    [PSCustomObject]$ffmodels_data
    [PSCustomObject]$slide_data
    [PSCustomObject]$slide_local_data
    [PSCustomObject]$sampledef_data
    [PSCustomObject]$sampledef_local_data
    [PSCustomObject]$worker_data
    [PSCustomObject]$mergeconfig_data
    [PSCustomObject]$imageqa_data
    [PSCustomObject]$storage_data
    [array]$projects
    #
    [array]$antibodies
    [array]$allprojects
    [hashtable]$allantibodies = @{}
    #
    [string]$cohorts_file = 'AstroPathCohortsProgress.csv' 
    [string]$paths_file = 'AstroPathPaths.csv'
    [string]$config_file = 'AstroPathConfig.csv'
    [string]$dependency_file = 'AstroPathDependencies.csv'
    [string]$slide_file = 'AstroPathAPIDdef.csv'
    [string]$slide_local_file = 'AstroPathAPIDdef'
    [string]$sampledef_file = 'AstroPathSampledef.csv'
    [string]$sampledef_local_file = 'sampledef.csv'
    [string]$ffmodels_file = 'AstroPathFlatfieldModels.csv' 
    [string]$corrmodels_file = 'AstroPathCorrectionModels.csv' 
    [string]$micomp_file = 'meanimagecomparison_table.csv' 
    [string]$worker_file = 'AstroPathHPFWLocs.csv'
    [string]$storage_file = 'AstroPathStorage.csv'
    [string]$imageqa_file = 'imageqa_upkeep.csv'
    [string]$imageqa_path = '\upkeep_and_progress'
    [string]$slide_local_path = 'upkeep_and_progress'
    [system.object]$mergefiles
    [system.object]$cantibodyfiles
    #
    [string]$onstrings = 'yes|y|on'
    [array]$logfileheaders = @('Project','Cohort','slideid','Message','Date')
    #
    [array]$imageqa_headers = @('comments')
    #
    [string]$apfile_constant = '.csv'
    #
    [string]$log_error = 'ERROR'
    [string]$log_start = 'START'
    [string]$log_finish = 'FINISH'
    #
    [string]$node_idle = 'IDLE'
    #
    $color_hex = @{
        Red = 'D10000'
        R = 'D10000'
        Green = '11FF00'
        G = '11FF00'
        Blue = '888888'
        B = '888888'
        Cyan = '0077FF'
        C = '0077FF'
        Yellow = 'FFF300'
        Y = 'FFF300'
        Magenta = 'FF00FF'
        M = 'FF00FF'
        White = 'FFFFFF'
        W = 'FFFFFF'
        Black = '000000'
        K = '000000'
        Orange = '923D00'
        O = '923D00'
    }
    #
    [string]apfullname($mpath, $file){
        return ($mpath + '\' + $file)
    }
    #
    [string]cohorts_fullfile($mpath){
        return $this.apfullname($mpath, $this.cohorts_file)
    }
    #
    [string]paths_fullfile($mpath){
        return $this.apfullname($mpath, $this.paths_file)
    }
    #
    [string]config_fullfile($mpath){
        return $this.apfullname($mpath, $this.config_file)
    }
    #
    [string]dependency_fullfile($mpath){
        return $this.apfullname($mpath, $this.dependency_file)
    }
    #
    [string]slide_fullfile($mpath){
        return $this.apfullname($mpath, $this.slide_file)
    }
    #
    [string]slide_local_fullfile($mpath){
        return ($mpath, $this.slide_local_path, $this.slide_local_file -join '\')
    }
    #
    [string]sampledef_fullfile($mpath){
        return $this.apfullname($mpath, $this.sampledef_file)
    }
    #
    [string]sampledef_local_fullfile($mpath){
        return $this.apfullname($mpath, $this.sampledef_local_file)
    }
    #
    [string]ffmodels_fullfile($mpath){
        return $this.apfullname($mpath, $this.ffmodels_file)
    }
    #
    [string]corrmodels_fullfile($mpath){
        return $this.apfullname($mpath, $this.corrmodels_file)
    }
    #
    [string]micomp_fullfile($mpath){
        return ($mpath + '\meanimagecomparison\' + $this.micomp_file)
    }
    #
    [string]worker_fullfile($mpath){
        return $this.apfullname($mpath, $this.worker_file)
    }
    #
    [string]storage_fullfile($mpath){
        return $this.apfullname($mpath, $this.storage_file)
    }
    #
    importaptables(){
        $this.importaptables($this.mpath)
    }
    #
    importaptables($mpath){
        $this.importcohortsinfo($mpath)
        $this.importdependencyinfo($mpath) 
        $this.importslideids($mpath) 
        $this.ImportFlatfieldModels($mpath) 
        $this.ImportCorrectionModels($mpath) 
        $this.ImportMICOMP($mpath) 
        $this.Importworkerlist($mpath) 
        $this.findallantibodies()
    }
    #
    importaptables($mpath, $createwatcher){
        $this.importcohortsinfo($mpath, $createwatcher) 
        $this.importdependencyinfo($mpath, $createwatcher) 
        $this.importslideids($mpath, $createwatcher) 
        $this.ImportFlatfieldModels($mpath, $createwatcher)
        $this.ImportCorrectionModels($mpath, $createwatcher) 
        $this.ImportMICOMP($mpath, $createwatcher) 
        $this.Importworkerlist($mpath, $createwatcher)
        $this.findallantibodies($false) ##### no file watchers on merge files!!!!
    }
    <# -----------------------------------------
     ImportCohortsInfo
     open the cohort info for the astropath
     processing pipeline with error checking from 
     the AstropathCohortsProgress.csv and 
     AstropathPaths.csv files in the mpath location
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportCohortsInfo(mpath)
    ----------------------------------------- #>
    [void]ImportCohortsInfo([string] $mpath, $createwatcher){
        #
        $cohort_csv_file = $this.cohorts_fullfile($mpath)
        #
        $cohort_data = $this.OpencsvFileConfirm($cohort_csv_file)
        #
        if ($createwatcher){
            $this.FileWatcher($cohort_csv_file)
        }
        #
        $paths_csv_file = $this.paths_fullfile($mpath)
        #
        if ($createwatcher){
            $this.FileWatcher($paths_csv_file)
        }
        #
        $paths_data = $this.OpencsvFileConfirm($paths_csv_file)
        #
        $config_csv_file = $this.config_fullfile($mpath)
        #
        if ($createwatcher){
            $this.FileWatcher($config_csv_file)
        }
        #
        $config_data = $this.OpencsvFileConfirm($config_csv_file)
        #
        $merged_project_dat = $this.MergeCustomObject(
            $cohort_data, $paths_data, 'Project')
        #
        $this.full_project_dat = $this.MergeCustomObject(
            $merged_project_dat, $config_data, 'Project')
        #
        $this.defallprojects()
        #
    }
    #
    [void]defallprojects(){
        #
        $this.allprojects = @()
        #
        if ($this.projects){
            $this.allprojects = $this.projects
        } else{
            $this.full_project_dat | ForEach-Object{
                $p = $this.uncpaths(($_.dpath, $_.dname -join '\'))
                if (test-path $p){
                    $this.allprojects += $_.project
                }
            }
        }
    }
    #
    [void]ImportCohortsInfo([string] $mpath){
        #
        if(!$this.full_project_dat){
            $this.importcohortsinfo($mpath, $false) 
        }
        #
    }
    #
    [void]ImportCohortsInfo(){
        #
        $this.importcohortsinfo($this.mpath) 
        #
    }
    <# -----------------------------------------
     ImportDependencyInfo
     open the config info for the astropath
     processing pipeline with error checking from 
     the AstroPathDependency.csv in the mpath location
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportDependencyInfo(mpath)
    ----------------------------------------- #>
    [void]ImportDependencyInfo([string] $mpath, $createwatcher){
        #
        $dependency_csv_file = $this.dependency_fullfile($mpath)
        $this.dependency_data = $this.OpencsvFileConfirm($dependency_csv_file)
        if ($createwatcher){
            $this.FileWatcher($dependency_csv_file)
        }
        #
    }
    #
    [void]ImportDependencyInfo([string] $mpath){
        #
        if(!$this.dependency_data){
            $this.ImportDependencyInfo($mpath, $false) 
        }
        #
    }
    #
    [void]ImportDependencyInfo(){
        #
        $this.ImportDependencyInfo($this.mpath)
        #
    }
    <# -----------------------------------------
     ImportSlideIDs
     open the AstropathAPIDdef.csv to get all slide
     available for processing
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportSlideIDs(mpath)
    ----------------------------------------- #>
    [void]ImportSlideIDs([string] $mpath, $createwatcher){
        #
        $this.importcohortsinfo($mpath)
        #
        $defpath = $this.slide_fullfile($mpath)
        $this.slide_data = $this.OpencsvFile($defpath)
        $this.slide_data = $this.slide_data | & { process {
            if (
                $_.project -match ($this.matcharray($this.allprojects))
            ){$_}
        }}
        #
        if ($createwatcher){
            $this.FileWatcher($defpath)
        }
        #
    }
    #
    [void]ImportSlideIDs([string] $mpath){  
        #
        if(!$this.slide_data){
            $this.ImportSlideIDs($mpath, $false) 
        }
        #
    }
    #
    [void]ImportSlideIDs(){
        #
        $this.ImportSlideIDs($this.mpath)
        #
    }
    #
    [void]ImportSlide($mpath){
        #
        if ($this.slide_data){
            return
        }
        #
        $this.slide_data = $this.OpencsvFile(
            $this.slide_fullfile($mpath)
        )
        #
    }
    #
    [void]ImportSlide($mpath, $createwatcher){
        #
        $this.importslideids($mpath, $createwatcher)
        #
    }
    #
    [void]Importsampledef([string] $mpath, $createwatcher){
        #
        $this.importcohortsinfo($mpath)
        #
        $defpath = $this.sampledef_fullfile($mpath)
        $this.sampledef_data = $this.OpencsvFile($defpath)
        $this.sampledef_data = $this.sampledef_data | & { process {
            if (
                $_.project -match ($this.matcharray($this.allprojects))
            ){$_}
        }}
        #
        if ($createwatcher){
            $this.FileWatcher($defpath)
        }
        #
    }
    #
    [void]Importsampledef([string] $mpath){  
        #
        if(!$this.sampledef_data){
            $this.Importsampledef($mpath, $false) 
        }
        #
    }
    #
    [void]Importsampledef(){
        #
        $this.Importsampledef($this.mpath)
        #
    }
    #
    [void]Importsampledef_local([string] $mpath, $createwatcher){
        #
        $this.importcohortsinfo($mpath)
        #
        $defpath = $this.sampledef_local_fullfile($mpath)
        $this.sampledef_local_data = $this.OpencsvFile($defpath)
        $this.sampledef_local_data = $this.sampledef_local_data | & { process {
            if (
                $_.project -match ($this.matcharray($this.allprojects))
            ){$_}
        }}
        #
        if ($createwatcher){
            $this.FileWatcher($defpath)
        }
        #
    }
    #
    [void]Importsampledef_local([string] $mpath){  
        #
        if(!$this.sampledef_local_data){
            $this.importsampledef_local($mpath, $false) 
        }
        #
    }
    #
    [void]Importsampledef_local(){
        #
        $this.importsampledef_local($this.basepath)
        #
    }
    #
    [void]Importslideids_local([string] $mpath, $createwatcher){
        #
        $this.importcohortsinfo($mpath)
        #
        $defpath = $this.slide_local_fullfile($mpath)
        $this.slide_local_data = $this.OpencsvFile($defpath)
        $this.slide_local_data = $this.slide_local_data | & { process {
            if (
                $_.project -match ($this.matcharray($this.allprojects))
            ){$_}
        }}
        #
        if ($createwatcher){
            $this.FileWatcher($defpath)
        }
        #
    }
    #
    [void]Importslideids_local([string] $mpath){  
        #
        if(!$this.slide_local_data){
            $this.Importslideids_local($mpath, $false) 
        }
        #
    }
    #
    [void]Importslideids_local(){
        #
        $this.Importslideids_local($this.basepath)
        #
    }
    <# -----------------------------------------
     ImportFlatfieldModels
     open the AstropathAPIDdef.csv to get all slide
     available for processing
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportFlatfieldModels(mpath)
    ----------------------------------------- #>
    [void]ImportFlatfieldModels([string] $mpath, $createwatcher){
        #
        $defpath = $this.ffmodels_fullfile($mpath)
        $this.ffmodels_data = $this.opencsvfile($defpath)
        if ($createwatcher){
            $this.FileWatcher($defpath)
        }
        #
     }
     #
    [void]ImportFlatfieldModels([string] $mpath){
        #
        if(!$this.ffmodels_data){
            $this.ImportFlatfieldModels($mpath, $false) 
        }
        #
     }
     #
    [void]Importffmodels($mpath, $createwatcher){
        #
        $this.ImportFlatfieldModels($mpath, $createwatcher)
        #
    }
    <# -----------------------------------------
     GetAPProjects
     Select the projects from the import config
     info 
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: GetAPProjects(mpath, module, project)
     Usage: GetAPProjects()
    ----------------------------------------- #>
    #
    [PSCustomObject]GetAPProjects(){
        #
        if (!$this.project){
            $cprojects = $this.getapprojects($this.module)
        } else {
            $cprojects = $this.project
        }
        return $cprojects
        #
     } 
    #
    [PSCustomObject]GetAPProjects($module){
        #
        $this.ImportCohortsInfo($this.mpath) 
        $headers = $this.gettablenames($this.full_project_dat)
        #
        if ($module -match $this.matcharray($headers)){
            $cprojects = ($this.full_project_dat | & { process {
                if (
                    $_.($module) -match $this.onstrings
                ) {$_}
            }}).Project
        } else {
            $cprojects = $this.full_project_dat.Project
        }
        #
        return $cprojects
        #
     }
    <# -----------------------------------------
     GetProjectCohortInfo
     Select the cohort info for a particular project
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: GetProjectCohortInfo(mpath, module, project)
     Usage: GetProjectCohortInfo()
    ----------------------------------------- #>
    [PSCustomObject]GetProjectCohortInfo([string] $mpath, [string] $project){
        #
        $this.ImportCohortsInfo($mpath)
        #
        $cleaned_project_dat = $this.full_project_dat | & { process {
            if ( $project -contains $_.Project ) { $_ }
        }}
        #
        return $cleaned_project_dat
        #
     }
    <# -----------------------------------------
     ImportCorrectionModels
     open the AstroPathCorrectionModels info for the astropath
     processing pipeline with error checking from 
     the AstroPathCorrectionModels.csv in the mpath location
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportCorrectionModels(mpath)
    ----------------------------------------- #>
    [void]ImportCorrectionModels([string] $mpath, $createwatcher){
        #
        $corr_csv_file = $this.corrmodels_fullfile($mpath)
        $this.corrmodels_data = $this.opencsvfile($corr_csv_file)
        if ($createwatcher){
            $this.FileWatcher($corr_csv_file)
        }
        #
    }
    #
    [void]ImportCorrectionModels([string] $mpath){
        #
        if (!$this.corrmodels_data){
            $this.ImportCorrectionModels($mpath, $false)
        }
        #
    }
    #
    [void]Importcorrmodels($mpath, $createwatcher){
        #
        $this.ImportCorrectionModels($mpath, $createwatcher)
        #
    }
    <# -----------------------------------------
     ImportMICOMP
     open the AstroPathmeanimagecomparison info 
     for the astropath processing pipeline with 
     error checking in the mpath location
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportMICOMP(mpath)
    ----------------------------------------- #>
    #
    [void]ImportMICOMP([string] $mpath, $createwatcher){
        #
        $micomp_csv_file = $this.micomp_fullfile($mpath)
        $this.micomp_data = $this.opencsvfile($micomp_csv_file)
        if ($createwatcher){
            $this.FileWatcher($micomp_csv_file)
        }
        #
    }
    #
    [void]ImportMICOMP([string] $mpath){
        #
        if (!$this.micomp_data){
            $this.importmicomp($mpath, $false)
        }
        #
    }
    #
    [string]mergeconfig_fullfile($basepath){
        #
        $mergefile = get-childitem ($basepath + '\Batch\*') "MergeConfig*xlsx"
        #
        if (!$mergefile){
            Throw ('merge config file could not be found for ' + $basepath)
        }
        #
        return $mergefile[0].fullname
        #
    }
    #
    [string]mergeconfig_fullfile($basepath, $batch){
        #
        $batchid = $batch.padleft(2,'0')
        return "$basepath\Batch\MergeConfig_$batchid.xlsx"
        #
    }
    #
    [string]mergeconfig_noerror_fullfile($basepath){
        #
        $mergefile = get-childitem ($basepath + '\Batch\*') "MergeConfig*xlsx"
        #
        if (!$mergefile){
            return @()
        }
        #
        return $mergefile[0].fullname
        #
    }
    #
    [string]mergeconfig_spath_fullfile($basepath){
        #
        $mergefile = get-childitem ($basepath + '\Batch\*') "MergeConfig*xlsx"
        #
        if (!$mergefile){
            Throw ('merge config file could not be found for basepath or spath: ' + $basepath)
        }
        #
        return $mergefile[0].fullname
        #
    }
    #
    [string]batch_fullfile($basepath){
        #
        $mergefile = get-childitem ($basepath + '\Batch\*') "Batch*xlsx"
        #
        if (!$mergefile){
            Throw ('batch file could not be found for ' + $basepath)
        }
        #
        return $mergefile[0].fullname
        #
    }
    #
    [string]batch_fullfile($basepath, $batch){
        #
        $batchid = $batch.padleft(2,'0')
        return "$basepath\Batch\BatchID_$batchid.xlsx"
        #
    }
    #
    [string]batch_noerror_fullfile($basepath){
        #
        $mergefile = get-childitem ($basepath + '\Batch\*') "Batch*xlsx"
        #
        if (!$mergefile){
            return @()
        }
        #
        return $mergefile[0].fullname
        #
    }
    #
    [string]batch_spath_fullfile($basepath){
        #
        $mergefile = get-childitem ($basepath + '\Batch\*') "Batch*xlsx"
        #
        if (!$mergefile){
            Throw ('batch file could not be found for basepath or spath:' + $basepath)
        }
        #
        return $mergefile[0].fullname
        #
    }
    #
    [string]mergeconfigcsv_fullfile($basepath){
        #
        $mergefile = get-childitem ($basepath + '\Batch\*') "MergeConfig*csv"
        #
        if (!$mergefile){
            Throw ('merge config file could not be found for ' + $basepath)
        }
        #
        return $mergefile[0].fullname
        #
    }
    #
    [string]mergeconfigcsv_fullfile($basepath, $batch){
        #
        $batchid = $batch.padleft(2,'0')
        return "$basepath\Batch\MergeConfig_$batchid.csv"
        #
    }
    #
    [void]ImportMergeConfig([string] $basepath, $createwatcher){
        #
        $micomp_csv_file = $this.mergeconfig_fullfile($basepath)
        $this.mergeconfig_data = $this.importexcel($this.mergeconfig_fullfile($basepath))
        if ($createwatcher){
            $this.FileWatcher($micomp_csv_file)
        }
        #
    }
    #
    [void]ImportMergeConfig([string] $basepath){
        #
        if (!$this.mergeconfig_data){
            $this.ImportMergeConfig($basepath, $false) 
        }
        #
    }
    #
    [void]ImportMergeConfig(){
        #
        $this.ImportMergeConfig($this.basepath) 
        #
    }
    #
    [void]findallantibodies(){
        $this.findallantibodies($false)
    }
    #
    [void]findallantibodies($createwatcher){
        #
        $this.full_project_dat | & { process{
            $p = $this.uncpaths(($_.dpath, $_.dname -join '\'))
            if (test-path $p){
                #
                $this.checksourcemergeconfig($p, ($_.spath, $_.dname -join '\'))
                #
                $this.allantibodies.($_.project) = 
                    $this.findantibodies($p, $createwatcher)
            }
        }}
        #
    }
    #
    [void]MergeConfigToCSV($basepath, $batch, $project, $cohort){
        #
        $batchid = ([string]$batch).PadLeft(2, '0')
        #
        $csvfile = $this.mergeconfigcsv_fullfile($basepath, $batchid)
        #
        if (Test-Path $csvfile){
           return
        }
        #
        $this.checksourcemergeconfig($basepath, $batchid, $project)
        #
        $this.mergeconfig_data = $this.importexcel($this.mergeconfig_fullfile($basepath, $batch))
        #
        $mergeconfig = $this.mergeconfig_data
        $count = 1
        foreach ($row in $mergeconfig) {
            $row | Add-Member -NotePropertyName Project `
                -NotePropertyValue $project -PassThru
            $row | Add-Member -NotePropertyName Cohort  `
                -NotePropertyValue $cohort -PassThru
            $row | Add-Member -NotePropertyName layer `
                -NotePropertyValue $count -PassThru
            $row.Colors = $this.color_hex.($row.Colors)
            $count++
        }
        #
        $AFlayer = [PSCustomObject]@{
            Project = $project
            Cohort = $cohort
            BatchID = $batch
            layer = $count
            Opal = 'AF'
            Target = 'NA'
            Compartment = 'Nucleus'
            TargetType = 'NA'
            CoexpressionStatus = 'NA'
            SegmentationStatus = '0'
            SegmentationHierarchy = '0'
            NumberofSegmentations = '0'
            ImageQA = 'NA'
            Colors = $this.color_hex.('Black')
        }
        $mergeconfig += $AFlayer
        #
        $mergeconfig | 
            Select-Object -Property Project,Cohort,BatchID,layer, `
                Opal,Target,Compartment,TargetType,CoexpressionStatus, `
                SegmentationStatus,SegmentationHierarchy, `
                NumberofSegmentations,ImageQA,Colors | 
            Export-Csv -Path $csvfile -NoTypeInformation
    }
    #
    [void]MergeConfigToCSV($basepath, $batchid){
        #
        $this.MergeConfigToCSV(
            $basepath, $batchid, $this.project, $this.cohort
        )
        #
    }
    #
    [void]MergeConfigToCSV(){
        #
        $this.MergeConfigToCSV(
            $this.basepath, $this.batchid, $this.project, $this.cohort
        )
        #
    }
    #
    [void]checksourcemergeconfig($basepath, $spath){
        #
        if (!($this.mergeconfig_noerror_fullfile($basepath))){     
            $file = $this.mergeconfig_spath_fullfile($this.uncpaths($spath))
            $this.copy($file, ($basepath + '\Batch'))
        }
        #
        if (!($this.batch_noerror_fullfile($basepath))){  
            $file = $this.batch_spath_fullfile($this.uncpaths($spath))
            $this.copy($file, ($basepath + '\Batch'))
        }
        #   
    }
    #
    [void]checksourcemergeconfig($basepath, $batch, $project){
        #
        if (!(test-path $this.mergeconfig_fullfile($basepath, $batch))){
            #
            $mdat = ($this.full_project_dat |
                where-object {$_.project -contains $project})
            $spathbase = $this.uncpaths($mdat.spath + '\' + $mdat.dname)
            #
            $path = $this.mergeconfig_fullfile($spathbase, $batch)
            #
            if (test-path $path){
                $this.copy($path, ($basepath + '\Batch'))
            } else {
                Throw "no merge file for project: $project - batch: $batch"
            }
            #
        }
        #
        if (!(test-path $this.batch_fullfile($basepath, $batch))){
            #
            $mdat = ($this.full_project_dat |
                where-object {$_.project -contains $project})
            $spathbase = $this.uncpaths($mdat.spath + '\' + $mdat.dname)
            #
            $path = $this.batch_fullfile($spathbase, $batch)
            #
            if (test-path $path){
                $this.copy($path, ($basepath + '\Batch'))
            } else {
                Throw "no batch file for in base or spath for project: $project - batch: $batch"
            }
            #
        }
        #   
    }
    #
    [void]ImportMergeConfigCSV([string] $basepath){
        #
        $this.mergeconfig_data =  $this.opencsvfile(
            $this.mergeconfigcsv_fullfile($basepath))
    }
    #
    [void]ImportMergeConfigCSV([string] $basepath, $batch){
        #
        $this.mergeconfig_data = $this.opencsvfile(
            $this.mergeconfigcsv_fullfile($basepath, $batch)
        )
    }
    #
    [array]findantibodies($basepath, $createwatcher){
        #
        $this.ImportMergeConfig($basepath, $createwatcher)
        if (!$this.mergeconfig_data){
            throw ('no merge config file for: ' + $basepath)
        }
        $lintargets = $this.deflineagemarkers()
        $exprtargets = $this.defexpressionmarkers()
        #
        return ($lintargets + $exprtargets)
        #
    }
    #
    [void]findantibodies($basepath){
        #
        $this.mergeconfigtocsv()
        #
        $this.ImportMergeConfigCSV($basepath)
        $lintargets = $this.deflineagemarkers()
        $exprtargets = $this.defexpressionmarkers()
        #
        $this.antibodies = $lintargets + $exprtargets
        #
    }
    #
    [void]findantibodies(){
        $this.findantibodies($this.basepath)
    }
    #
    [void]getantibodies($project){
        #
        if (!($this.allantibodies.($project))){
            $p1 = $this.full_project_dat | Where-Object {$_.project -eq $project}
            $p = $this.uncpaths(($p1.dpath, $p1.dname -join '\'))
            $this.allantibodies.($project) = $this.findantibodies($p, $false)
        }
        #
        $this.antibodies = $this.allantibodies.($project)
        #
    }
    #
    [void]getantibodies(){
        #
        $this.mergeconfigtocsv()
        $this.getantibodies($this.project)
        #
    }
    #
    [array]deflineagemarkers(){
        #
        $data = $this.mergeconfig_data | & { process {
            if (
                $_.Opal -notcontains 'DAPI' -and
                $_.ImageQA -notmatch 'membrane' -and
                $_.TargetType -match 'Lineage'
            ) {$_}
        }}
        #
        if ($data){
            $lintargets = $data.Target
            $qa = $data.ImageQA.indexOf('Tumor')
            #
            if ($qa -ge 0){
                $lintargets[$qa] = 'Tumor'
            }
            return $lintargets
        } else {
            return @()
        }
        #
    }
    #
    [array]defexpressionmarkers(){
        #
        $data = $this.mergeconfig_data | & { process {
            if (
                $_.Opal -notcontains 'DAPI' -and
                $_.ImageQA -notmatch 'membrane' -and
                $_.TargetType -match 'Expression'
            ) {$_}
        }}
        #
        if ($data){
            $exprtargets = $data.Target
            #
            $multisegs = $data | Where-Object {
                $_.numberofsegmentations -gt 1
            }
            #
            foreach ($multiseg in $multisegs){
                $nextrasegs = $multiseg.numberofsegmentations
                2..$nextrasegs | foreach-object{
                    $exprtargets += ($multiseg.Target + '_' + [string]$_)
                }
            }
            #
            return $exprtargets
        } else {
            return @()
        }
        #
    }
    <# -----------------------------------------
     Importworkerlist
     import and return a log file object
     ------------------------------------------
     Input: 
        -fpath: full path to the log
     ------------------------------------------
     Usage: Importworkerlist($fpath)
    ----------------------------------------- #>
    #
    [void]Importworkerlist([string] $mpath, $createwatcher){
        #
        $tbl = $this.opencsvfileconfirm(
            $this.worker_fullfile($mpath)
        )
        #
        if ($this.worker_data){
            #
            foreach ($row in $tbl){
                #
                $oldrow = $this.worker_data | where-object {
                    $_.server -contains $row.server -and
                    $_.location -contains $row.location -and
                    $_.module -contains $row.module
                }
                #
                if ($oldrow){
                   $row | Add-Member status $oldrow.status -PassThru
                } else {
                    $row | Add-Member status 'IDLE' -PassThru
                }
                #
            }
            #
        } else {
            $tbl |
                Add-Member -NotePropertyName 'Status' -NotePropertyValue $this.node_idle
        }
        #
        $this.worker_data = $tbl
        #
        if ($createwatcher){
            $this.FileWatcher($this.worker_fullfile($mpath))
        }
        #
    }
    #
    [void]Importworkerlist([string] $mpath){
        #
        if (!$this.worker_data){
            $this.Importworkerlist($mpath, $false) 
        }
        #
    }
    #
    [string]imageqa_fullpath(){
        return $this.imageqa_fullpath($this.basepath)
    }
    #
    [string]imageqa_fullpath($basepath){
        $imageqa_filepath = $basepath + 
            $this.imageqa_path + '\' + $this.imageqa_file
        return $imageqa_filepath
    }
    #
    [array]buildimageqaheaders($cantibodies){
        #
        $str = @('SlideID')
        $cantibodies | ForEach-Object{
            $str += $_
        }
        $headers = $str + $this.imageqa_headers
        #
        return $headers
        #
    }
    #
    [void]ImportImageQA(){
        #
        $this.ImportImageQA($this.basepath)
        #
    }
    #
    [void]ImportImageQA($basepath){
        #
        if (!$this.antibodies){
            $this.findantibodies()
        }
        #
        $cantibodies = $this.antibodies
        $this.ImportImageQA($basepath, $cantibodies)
        #
    }
    #
    [void]ImportImageQA($basepath, $cantibodies){
        #
        $this.imageqa_data = $this.opencsvfile(
            $this.imageqa_fullpath($basepath), 
            $this.buildimageqaheaders($cantibodies))
        #
    }
    #
    [void]AddImageQA($basepath, $slideid, $cantibodies){
        #
        $str = $slideid
        $cantibodies | ForEach-Object{
            $str += ','
        }
        #
        $this.imageqa_headers | ForEach-Object{
            $str += ','
        }
        #
        $str += "`r`n"
        #
        $this.popfile($this.imageqa_fullpath($basepath), $str)
        #
    }
    #
    [void]removelogwatcher($module, $project){
        #
        $fpath = $this.defprojectlogpath($module, $project)
        try{
            $this.UnregisterEvent($fpath)
        } catch {
            Write-Host 'ERROR removing watcher on:' $fpath
        }
        #
     }
     #
     [string]defprojectlogpath($module, $project){
            #
            $this.importcohortsinfo()
            $project_dat = $this.full_project_dat | & { process {
                if ($_.project -contains $project){ $_ }
            }}
            #
            return( 
                $this.uncpaths($project_dat.dpath),
                $project_dat.dname, 'logfiles',
                ($module,'.log' -join '') -join '\'
            )
            #
     }
     <# -----------------------------------------
     Importlogfile
     import and return a log file object
     ------------------------------------------
     Input: 
        -fpath: full path to the log
     ------------------------------------------
     Usage: Importlogfile($fpath)
    ----------------------------------------- #>
    #
    [PSCustomObject]Importlogfile($module, $project, $createwatcher){
        #
        $fpath = $this.defprojectlogpath($module, $project)
        $logfile = $this.importlogfile($fpath)
        #
        if ($createwatcher){
            $this.FileWatcher($fpath)
        }
        #
        return $logfile
        #
     }
     #
    [PSCustomObject]Importlogfile($module, $project){
        #
        return $this.importlogfile(
            $this.defprojectlogpath($module, $project)
        )
        #
    }
    #
    [PSCustomObject]Importlogfile([string] $fpath){
        #
        if (test-path $fpath){
            $logfile = $this.opencsvfile($fpath, `
                ';', $this.logfileheaders)
        } else {
            $this.createfile($fpath)
            $logfile = ''
        }
        #
        return $logfile
        #
    }
    #
    [void]UpdateStorage($mpath){
        #
        $storage = Get-PSDrive | 
            Where-Object {$_.Provider -match 'FileSystem'} | 
            Select-Object -Property Name, Root, Free
        Add-Member -InputObject $storage -MemberType NoteProperty -Name "Description" -Value ""
        $storage | ForEach-Object{$_.Free /= 1TB}
        $this.storage_data = $storage | 
            Select-Object @{N='Server';E={$_.Root}}, 
                          @{N='Drive';E={$_.Name}}, 
                          @{N='Description';E={$_.Description}},
                          @{N='Space TB';E={$_.Free}}
        $this.storage_data | Export-Csv -Path $this.storage_fullfile($mpath) -NoTypeInformation
        #
    }
    #
    <# -----------------------------------------
     selectlogline
     select the most recent line for the input
     type
     ------------------------------------------
     Input: 
        - loglines: the log itself
        - ID: the log entry type to match (batch or slideid)
        - status: the status to match (ERROR, START, FINISH, WARNING)
        - vers: the version number to match
        - [antibody]: the antibody to match for vminform
        - [algorithm]: the algorithm to match for vminform
     ------------------------------------------
     Usage: selectlogline($fpath)
    ----------------------------------------- #>
    [PSCustomObject]selectloglines([PSCustomObject] $loglines,
     [string] $ID){    
        #
        return ( $loglines | &{ process {
                if (
                    $_.Slideid -contains $ID
                ) { $_ }
            }}
        )
        #
    }
    #
    [PSCustomObject]selectlogline([PSCustomObject] $loglines,
     [string] $ID, [string] $status){    
        #
        return ( $loglines | &{ process {
                if (
                    ($_.Slideid -contains $ID) -and 
                    ($_.Message -match ('^' + $status))
                ) { $_ }
            }} |
            Select-Object -Last 1
        )
        #
    }
    #
    [PSCustomObject]selectlogline([PSCustomObject] $loglines,
     [string] $ID, [string] $status, [string] $vers){    
        #
        return (
            $loglines | &{ process {
            if (
                ($_.Message -match $vers) -and 
                ($_.Slideid -contains $ID) -and 
                ($_.Message -match ('^' + $status))
                ) { $_ }
            }} |
            Select-Object -Last 1
        )
        #
    }
    #
    [PSCustomObject]selectlogline([PSCustomObject] $loglines,
     [string] $ID, [string] $status, [string] $vers, [string] $antibody){    
        #
        return ( $loglines | &{ process {
            if (
                ($_.Slideid -contains $ID) -and 
                ($_.Message -match ('^' + $status)) -and 
                ($_.Message -match ('Antibody: ' +
                    $antibody + ' - Algorithm:'))
                ) { $_ }
            }} |
            Select-Object -Last 1
        )
        #
    } 
    #
    [PSCustomObject]selectlogline([PSCustomObject] $loglines,
     [string] $ID, [string] $status, [string] $vers, [string] $antibody,
     [string] $algorithm){    
        #
        return  (
            $loglines | &{ process {
            if (
                ($_.Slideid -contains $ID) -and 
                ($_.Message -match ('^' + $status)) -and 
                ($_.Message -match ('Antibody: ' +
                    $antibody + ' - Algorithm: ' + $algorithm))
            ) { $_ }
            }} |
            Select-Object -Last 1
        )
        #
    } 
    # 
}