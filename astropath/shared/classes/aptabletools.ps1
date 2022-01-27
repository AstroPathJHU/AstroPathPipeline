class aptabletools : fileutils {
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
    [PSCustomObject]ImportCohortsInfo([string] $mpath){
        #
        $cohort_csv_file = $mpath + '\AstropathCohortsProgress.csv'
        #
        $project_data = $this.OpencsvFileConfirm($cohort_csv_file)
        #
        $paths_csv_file = $mpath + '\AstropathPaths.csv'
        #
        $paths_data = $this.OpencsvFileConfirm($paths_csv_file)
        #
        $project_data = $this.MergeCustomObject( $project_data, $paths_data, 'Project')
        #
        return $project_data
        #
    }
    <# -----------------------------------------
     ImportConfigInfo
     open the config info for the astropath
     processing pipeline with error checking from 
     the AstropathConfig.csv in the mpath location
     ------------------------------------------
     Input: 
        -mpath: main path for the astropath processing
         which contains all necessary processing files
     ------------------------------------------
     Usage: ImportCohortsInfo(mpath)
    ----------------------------------------- #>
    [PSCustomObject]ImportConfigInfo([string] $mpath){
        #
        $config_csv_file = $mpath + '\AstropathConfig.csv'
        #
        $config_data = $this.OpencsvFileConfirm($config_csv_file)
        #
        return $config_data
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
    [PSCustomObject]ImportSlideIDs([string] $mpath){
        #
        $defpath = $mpath + '\AstropathAPIDdef.csv'
        #
        $slide_ids = $this.opencsvfile($defpath)
        return $slide_ids
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
    [PSCustomObject]ImportFlatfieldModels([string] $mpath){
        #
        $defpath = $mpath + '\AstroPathFlatfieldModels.csv'
        #
        $slide_ids = $this.opencsvfile($defpath)
        return $slide_ids
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
    [PSCustomObject]GetAPProjects([string] $mpath, [string] $module, [string] $project){
        #
        $project_dat = $this.ImportConfigInfo($mpath)
        #
        if (!$project){
            $projects = ($project_dat | 
                Where-object {$_.($module) -match 'yes'}).Project
        } else {
            $projects = $project
        }
        return $projects
        #
     }
    #
    [PSCustomObject]GetAPProjects(){
        #
        $project_dat = $this.ImportConfigInfo($this.mpath)
        #
        if (!$this.project){
            $projects = ($project_dat | 
                Where-object {$_.($this.module) -match 'yes'}).Project
        } else {
            $projects = $this.project
        }
        return $projects
        #
     } 
    #
    [PSCustomObject]GetAPProjects($module){
        #
        $project_dat = $this.ImportConfigInfo($this.mpath)
        #
        if (!$this.project){
            $projects = ($project_dat | 
                Where-object {$_.($module) -match 'yes'}).Project
        } else {
            $projects = $this.project
        }
        return $projects
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
     Usage: GetAPProjects(mpath, module, project)
     Usage: GetAPProjects()
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
     Usage: ImportCohortsInfo(mpath)
    ----------------------------------------- #>
    [PSCustomObject]ImportCorrectionModels([string] $mpath){
        #
        $config_csv_file = $mpath + '\AstroPathCorrectionModels.csv'
        #
        $config_data = $this.opencsvfile($config_csv_file)
        #
        return $config_data
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
    [PSCustomObject]Importlogfile([string] $fpath){
        #
        $logfile = $this.opencsvfile($fpath, `
            ';', @('Project','Cohort','slideid','Message','Date'))
        #
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
     Usage: Importlogfile($fpath)
    ----------------------------------------- #>
    [PSCustomObject]selectlogline([PSCustomObject] $loglines, [string] $ID, [string] $status, [string] $vers){    
        #
        $logline = $loglines |
                where-object {
                    ($_.Message -match $vers) -and 
                        ($_.Slideid -match $ID) -and 
                        ($_.Message -match $status)
                } |
                Select-Object -Last 1
        #
        return $logline
        #
    }
    #
    [PSCustomObject]selectlogline([PSCustomObject] $loglines, [string] $ID, [string] $status, [string] $vers, [string] $antibody){    
        #
        $logline = $loglines |
                where-object {
                    ($_.Slideid -match $ID) -and 
                        ($_.Message -match $status) -and 
                        ($_.Message -match ('Antibody: ' + $antibody + ' - Algorithm:'))
                } |
                Select-Object -Last 1
        #
        return $logline
        #
    } 
    #
    [PSCustomObject]selectlogline([PSCustomObject] $loglines, [string] $ID, [string] $status, [string] $vers, [string] $antibody, [string] $algorithm){    
        #
        $logline = $loglines |
                where-object {
                    ($_.Slideid -match $ID) -and 
                        ($_.Message -match $status) -and 
                        ($_.Message -match ('Antibody: ' + $antibody + ' - Algorithm: ' + $algorithm))
                } |
                Select-Object -Last 1
        #
        return $logline
        #
    } 
    #        
}