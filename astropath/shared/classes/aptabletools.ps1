class aptabletools : fileutils {
    #
    [PSCustomObject]$full_project_dat
    [PSCustomObject]$dependency_data
    [PSCustomObject]$micomp_data
    [PSCustomObject]$corrmodels_data
    [PSCustomObject]$ffmodels_data
    [PSCustomObject]$slide_data
    [PSCustomObject]$worker_data
    [PSCustomObject]$mergeconfig_data
    [PSCustomObject]$imageqa_data
    [array]$projects
    #
    [array]$antibodies
    [array]$allprojects
    #
    [string]$cohorts_file = 'AstroPathCohortsProgress.csv' 
    [string]$paths_file = 'AstroPathPaths.csv'
    [string]$config_file = 'AstroPathConfig.csv'
    [string]$dependency_file = 'AstroPathDependencies.csv'
    [string]$slide_file = 'AstroPathAPIDdef.csv'
    [string]$ffmodels_file = 'AstroPathFlatfieldModels.csv' 
    [string]$corrmodels_file = 'AstroPathCorrectionModels.csv' 
    [string]$micomp_file = 'meanimagecomparison_table.csv' 
    [string]$worker_file = 'AstroPathHPFWLocs.csv' 
    [string]$imageqa_file = 'imageqa_upkeep.csv'
    [string]$imageqa_path = '\upkeep_and_progress'
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
    importaptables(){
        $this.importaptables($this.mpath)
    }
    #
    importaptables($mpath){
        $this.importcohortsinfo($mpath) | Out-Null
        $this.importdependencyinfo($mpath) | Out-Null
        $this.importslideids($mpath) | Out-Null
        $this.ImportFlatfieldModels($mpath) | Out-Null
        $this.ImportCorrectionModels($mpath) | Out-Null
        $this.ImportMICOMP($mpath) | Out-Null
        $this.Importworkerlist($mpath) | Out-Null
    }
    #
    importaptables($mpath, $createwatcher){
        $this.importcohortsinfo($mpath, $createwatcher) | Out-Null
        $this.importdependencyinfo($mpath, $createwatcher) | Out-Null
        $this.importslideids($mpath, $createwatcher) | Out-Null
        $this.ImportFlatfieldModels($mpath, $createwatcher) | Out-Null
        $this.ImportCorrectionModels($mpath, $createwatcher) | Out-Null
        $this.ImportMICOMP($mpath, $createwatcher) | Out-Null
        $this.Importworkerlist($mpath, $createwatcher) | Out-Null
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
    [PSCustomObject]ImportCohortsInfo([string] $mpath, $createwatcher){
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
        return $this.full_project_dat
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
    [PSCustomObject]ImportCohortsInfo([string] $mpath){
        #
        if(!$this.full_project_dat){
            $this.importcohortsinfo($mpath, $false) | Out-NULL
        }
        #
        return $this.full_project_dat
        #
    }
    #
    [PSCustomObject]ImportCohortsInfo(){
        #
         return $this.importcohortsinfo($this.mpath) 
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
    [PSCustomObject]ImportDependencyInfo([string] $mpath, $createwatcher){
        #
        $dependency_csv_file = $this.dependency_fullfile($mpath)
        $this.dependency_data = $this.OpencsvFileConfirm($dependency_csv_file)
        if ($createwatcher){
            $this.FileWatcher($dependency_csv_file)
        }
        return $this.dependency_data
        #
    }
    #
    [PSCustomObject]ImportDependencyInfo([string] $mpath){
        #
        if(!$this.dependency_data){
            $this.ImportDependencyInfo($mpath, $false) | Out-Null
        }
        return $this.dependency_data
        #
    }
    #
    [PSCustomObject]ImportDependencyInfo(){
        #
        return $this.ImportDependencyInfo($this.mpath)
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
    [PSCustomObject]ImportSlideIDs([string] $mpath, $createwatcher){
        #
        $defpath = $this.slide_fullfile($mpath)
        $this.slide_data = $this.OpencsvFileConfirm($defpath)
        if ($createwatcher){
            $this.FileWatcher($defpath)
        }
        return $this.slide_data 
        #
    }
    #
    [PSCustomObject]ImportSlideIDs([string] $mpath){
        #
        if(!$this.slide_data){
            $this.ImportSlideIDs($mpath, $false) | Out-Null
        }
        return $this.slide_data
        #
    }
    #
    [PSCustomObject]ImportSlideIDs(){
        #
        return $this.ImportSlideIDs($this.mpath)
        #
    }
    #
    [PSCustomObject]ImportSlide($mpath, $createwatcher){
        #
        return $this.ImportSlideIDs($mpath, $createwatcher)
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
    [PSCustomObject]ImportFlatfieldModels([string] $mpath, $createwatcher){
        #
        $defpath = $this.ffmodels_fullfile($mpath)
        $this.ffmodels_data = $this.opencsvfile($defpath)
        if ($createwatcher){
            $this.FileWatcher($defpath)
        }
        return $this.ffmodels_data
        #
     }
     #
    [PSCustomObject]ImportFlatfieldModels([string] $mpath){
        #
        if(!$this.ffmodels_data){
            $this.ImportFlatfieldModels($mpath, $false) | Out-NUll
        }
        return $this.ffmodels_data
        #
     }
     #
    [PSCustomObject]Importffmodels($mpath, $createwatcher){
        #
        return $this.ImportFlatfieldModels($mpath, $createwatcher)
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
        $this.ImportCohortsInfo($this.mpath) | out-null
        $headers = $this.gettablenames($this.full_project_dat)
        #
        if ($module -match ('^' + ($headers -join '$|^') + '$')){
            $cprojects = ($this.full_project_dat | 
                Where-object {$_.($module) -match $this.onstrings}).Project
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
        $project_dat = $this.ImportCohortsInfo($mpath)
        #
        $cleaned_project_dat = $project_dat | 
                Where-Object {$project -contains $_.Project}
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
    [PSCustomObject]ImportCorrectionModels([string] $mpath, $createwatcher){
        #
        $corr_csv_file = $this.corrmodels_fullfile($mpath)
        $this.corrmodels_data = $this.opencsvfile($corr_csv_file)
        if ($createwatcher){
            $this.FileWatcher($corr_csv_file)
        }
        #
        return $this.corrmodels_data 
        #
    }
    #
    [PSCustomObject]ImportCorrectionModels([string] $mpath){
        #
        if (!$this.corrmodels_data){
            $this.ImportCorrectionModels($mpath, $false) | Out-NULL
        }
        #
        return $this.corrmodels_data 
        #
    }
    #
    [PSCustomObject]Importcorrmodels($mpath, $createwatcher){
        #
        return $this.ImportCorrectionModels($mpath, $createwatcher)
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
    [PSCustomObject]ImportMICOMP([string] $mpath, $createwatcher){
        #
        $micomp_csv_file = $this.micomp_fullfile($mpath)
        $this.micomp_data = $this.opencsvfile($micomp_csv_file)
        if ($createwatcher){
            $this.FileWatcher($micomp_csv_file)
        }
        #
        return $this.micomp_data
        #
    }
    #
    [PSCustomObject]ImportMICOMP([string] $mpath){
        #
        if (!$this.micomp_data){
            $this.importmicomp($mpath, $false) | Out-NULL
        }
        #
        return $this.micomp_data
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
    [PSCustomObject]ImportMergeConfig([string] $basepath, $createwatcher){
        #
        $micomp_csv_file = $this.mergeconfig_fullfile($basepath)
        $this.mergeconfig_data = $this.importexcel($micomp_csv_file)
        if ($createwatcher){
            $this.FileWatcher($micomp_csv_file)
        }
        #
        return $this.mergeconfig_data
        #
    }
    #
    [PSCustomObject]ImportMergeConfig([string] $basepath){
        #
        if (!$this.mergeconfig_data){
            $this.ImportMergeConfig($basepath, $false) | Out-NULL
        }
        #
        return $this.mergeconfig_data
        #
    }
    #
    [PSCustomObject]ImportMergeConfig(){
        #
        $this.ImportMergeConfig($this.basepath) | Out-NULL
        #
        return $this.mergeconfig_data
        #
    }
    #
    [void]findantibodies(){
        $this.findantibodies($this.basepath)
    }
    #
    [void]findantibodies($basepath){
        #
        $this.ImportMergeConfig($basepath)
        $data = $this.mergeconfig_data | 
            Where-Object {$_.Opal -notcontains 'DAPI' `
                -and $_.Target -notcontains 'Membrane'}
        $targets = $data.Target
        $qa = $data.ImageQA.indexOf('Tumor')
        #
        if ($qa -ge 0){
            $targets[$qa] = 'Tumor'
        }
        #
        $this.antibodies = $targets
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
    [PSCustomObject]Importworkerlist([string] $mpath, $createwatcher){
        #
        $worker_csv_file = $this.worker_fullfile($mpath)
        $this.worker_data = $this.opencsvfileconfirm($worker_csv_file)
        $headers = $this.gettablenames($this.worker_data)
        if ('Status' -notmatch ($headers -join '|')){
            $this.worker_data |
                Add-Member -NotePropertyName 'Status' -NotePropertyValue $this.node_idle
        } else {
            $this.worker_data | 
                Where-Object {$_.Status -match $this.onstrings}
        }
        if ($createwatcher){
            $this.FileWatcher($worker_csv_file)
        }
        #
        return $this.worker_data
        #
    }
    #
    [PSCustomObject]Importworkerlist([string] $mpath){
        #
        if (!$this.worker_data){
            $this.Importworkerlist($mpath, $false) | Out-NULL
        }
        #
        return $this.worker_data
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
            $this.importcohortsinfo() | Out-Null
            $project_dat = $this.full_project_dat |
                Where-Object {$_.project -contains $project}
            #
            $root = $this.uncpaths($project_dat.dpath)
            $fpath = $root, $project_dat.dname, 'logfiles',
             ($module,'.log' -join '') -join '\'
            #
            return $fpath
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
        $logfile = $this.importlogfile(
            $this.defprojectlogpath($module, $project)
        )
        #
        return $logfile
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
    [PSCustomObject]selectlogline([PSCustomObject] $loglines,
     [string] $ID, [string] $status){    
        #
        $logline = $loglines |
                where-object {
                        ($_.Slideid -match $ID) -and 
                        ($_.Message -match ('^' + $status))
                } |
                Select-Object -Last 1
        #
        return $logline
        #
    }
    #
    [PSCustomObject]selectlogline([PSCustomObject] $loglines,
     [string] $ID, [string] $status, [string] $vers){    
        #
        $logline = $loglines |
                where-object {
                    ($_.Message -match $vers) -and 
                        ($_.Slideid -match $ID) -and 
                        ($_.Message -match ('^' + $status))
                } |
                Select-Object -Last 1
        #
        return $logline
        #
    }
    #
    [PSCustomObject]selectlogline([PSCustomObject] $loglines,
     [string] $ID, [string] $status, [string] $vers, [string] $antibody){    
        #
        $logline = $loglines |
                where-object {
                    ($_.Slideid -match $ID) -and 
                        ($_.Message -match ('^' + $status)) -and 
                        ($_.Message -match ('Antibody: ' +
                         $antibody + ' - Algorithm:'))
                } |
                Select-Object -Last 1
        #
        return $logline
        #
    } 
    #
    [PSCustomObject]selectlogline([PSCustomObject] $loglines,
     [string] $ID, [string] $status, [string] $vers, [string] $antibody,
     [string] $algorithm){    
        #
        $logline = $loglines |
                where-object {
                    ($_.Slideid -match $ID) -and 
                        ($_.Message -match ('^' + $status)) -and 
                        ($_.Message -match ('Antibody: ' +
                         $antibody + ' - Algorithm: ' + $algorithm))
                } |
                Select-Object -Last 1
        #
        return $logline
        #
    } 
    # 
}