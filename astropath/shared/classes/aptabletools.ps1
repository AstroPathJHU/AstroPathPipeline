class aptabletools : fileutils {
    #
    [PSCustomObject]$full_project_dat
    [PSCustomObject]$config_data
    [PSCustomObject]$micomp_data
    [PSCustomObject]$corrmodels_data
    [PSCustomObject]$ffmodels_data
    [PSCustomObject]$slide_data
    #
    importaptables($mpath){
        $this.importcohortsinfo($mpath)
        $this.importconfiginfo($mpath)
        $this.importslideids($mpath)
        $this.ImportFlatfieldModels($mpath)
        $this.ImportCorrectionModels($mpath)
        $this.ImportMICOMP($mpath)
    }
    #
    importaptables($mpath, $forceupdate){
        $this.importcohortsinfo($mpath, $forceupdate)
        $this.importconfiginfo($mpath, $forceupdate)
        $this.importslideids($mpath, $forceupdate)
        $this.ImportFlatfieldModels($mpath, $forceupdate)
        $this.ImportCorrectionModels($mpath, $forceupdate)
        $this.ImportMICOMP($mpath, $forceupdate)
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
    [PSCustomObject]ImportCohortsInfo([string] $mpath, $forceupdate){
        #
        $cohort_csv_file = $mpath + '\AstropathCohortsProgress.csv'
        #
        $project_data = $this.OpencsvFileConfirm($cohort_csv_file)
        #
        $paths_csv_file = $mpath + '\AstropathPaths.csv'
        #
        $paths_data = $this.OpencsvFileConfirm($paths_csv_file)
        #
        $this.full_project_dat = $this.MergeCustomObject( $project_data, $paths_data, 'Project')
        #
        return $this.full_project_dat
        #
    }
    #
    [PSCustomObject]ImportCohortsInfo([string] $mpath){
        #
        if(!$this.full_project_dat){
            $this.importcohortsinfo($mpath, $true) | Out-NULL
        }
        #
        return $this.full_project_dat
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
    [PSCustomObject]ImportConfigInfo([string] $mpath, $forceupdate){
        #
        $config_csv_file = $mpath + '\AstropathConfig.csv'
        $this.config_data = $this.OpencsvFileConfirm($config_csv_file)
        return $this.config_data
        #
    }
    #
    [PSCustomObject]ImportConfigInfo([string] $mpath){
        #
        if(!$this.config_data){
            $this.ImportConfigInfo($mpath, $true) | Out-Null
        }
        return $this.config_data
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
    [PSCustomObject]ImportSlideIDs([string] $mpath, $forceupdate){
        #
        $defpath = $mpath + '\AstropathAPIDdef.csv'
        $this.slide_data = $this.OpencsvFileConfirm($defpath)
        return $this.slide_data 
        #
    }
    #
    [PSCustomObject]ImportSlideIDs([string] $mpath){
        #
        if(!$this.slide_data){
            $this.ImportSlideIDs($mpath, $true) | Out-Null
        }
        return $this.slide_data
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
    [PSCustomObject]ImportFlatfieldModels([string] $mpath, $forceupdate){
        #
        $defpath = $mpath + '\AstroPathFlatfieldModels.csv'
        $this.ffmodels_data = $this.opencsvfile($defpath)
        return $this.ffmodels_data
        #
     }
     #
    [PSCustomObject]ImportFlatfieldModels([string] $mpath){
        #
        if(!$this.ffmodels_data){
            $this.ImportFlatfieldModels($mpath, $true) | Out-NUll
        }
        return $this.ffmodels_data
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
    [PSCustomObject]ImportCorrectionModels([string] $mpath, $forceupdate){
        #
        $config_csv_file = $mpath + '\AstroPathCorrectionModels.csv'
        $this.corrmodels_data = $this.opencsvfile($config_csv_file)
        #
        return $this.corrmodels_data 
        #
    }
    #
    [PSCustomObject]ImportCorrectionModels([string] $mpath){
        #
        if (!$this.corrmodels_data){
            $this.ImportCorrectionModels($mpath, $true) | Out-NULL
        }
        #
        return $this.corrmodels_data 
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
    [PSCustomObject]ImportMICOMP([string] $mpath, $forceupdate){
        #
        $micomp_csv_file = $mpath + '\meanimagecomparison\meanimagecomparison_table.csv'
        $this.micomp_data = $this.opencsvfile($micomp_csv_file)
        #
        return $this.micomp_data
        #
    }
    #
    [PSCustomObject]ImportMICOMP([string] $mpath){
        #
        if (!$this.micomp_data){
            $this.importmicomp($mpath, $true) | Out-NULL
        }
        #
        return $this.micomp_data
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
     Usage: selectlogline($fpath)
    ----------------------------------------- #>
    [PSCustomObject]selectlogline([PSCustomObject] $loglines, [string] $ID, [string] $status){    
        #
        $logline = $loglines |
                where-object {
                        ($_.Slideid -match $ID) -and 
                        ($_.Message -match $status)
                } |
                Select-Object -Last 1
        #
        return $logline
        #
    }
    #
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