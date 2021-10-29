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
        $project_data = $this.OpencsvFile($cohort_csv_file)
        #
        $paths_csv_file = $mpath + '\AstropathPaths.csv'
        #
        $paths_data = $this.opencsvfile($paths_csv_file)
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
        $config_data = $this.opencsvfile($config_csv_file)
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
        $slide_ids = $this.opencsvfile( $defpath)
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
        $slide_ids = $this.opencsvfile( $defpath)
        return $slide_ids
        #
     }
     #
}