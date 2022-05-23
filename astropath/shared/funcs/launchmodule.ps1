function LaunchModule{
    param(
        [Parameter()][string]$mpath='\\bki04\astropath_processing',
        [Parameter()][string]$module = '',
        [Parameter()][string]$project='',
        [Parameter()][string]$slideid='',
        [Parameter()][string]$batchid='',
        [Parameter()][string]$processloc='',
        [Parameter()][string]$taskid='',
        [Parameter()][string]$antibody='',
        [Parameter()][string]$algorithm='',
        [Parameter()][string]$informvers='',
        [Parameter()][string]$tasklogfile='',
        [Parameter()][switch]$test

    )
    #
    if ($PSBoundParameters.test){
        $inp = initmodule -task $PSBoundParameters -module $module -test    
        return $inp
    }
    #
    if ($module -match 'batch'){
        $m = [launchmodule]::new($mpath, $module,
            $batchid, $project, $PSBoundParameters)
        #
    } else {
        $m = [launchmodule]::new($mpath, $module,
            $slideid, $PSBoundParameters)
    }        
    #
    return $m
    #
}
#