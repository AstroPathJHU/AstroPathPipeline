function sampletracker {
    param(
        [parameter()][string]$mpath,
        [parameter()][vminformqueue]$vmq,
        [parameter()][array]$modules,
        [parameter()][hashtable]$modulelogs,
        [parameter()][string]$slideid
    )
    #
    if (!($PSBoundParameters.ContainsKey('vmq'))){
        return [sampletracker]::new($mpath)
    }
    #
    if (!($PSBoundParameters.ContainsKey('modules'))){
        return [sampletracker]::new($mpath, $vmq)
    }
    #
    if (!($PSBoundParameters.ContainsKey('modulelogs'))){
        return [sampletracker]::new($mpath, $vmq, $modules)
    }
    #
    if (!($PSBoundParameters.ContainsKey('slideid'))){
        return [sampletracker]::new($mpath, $vmq, $modules, $modulelogs)
    }
    #
    return [sampletracker]::new($mpath, $vmq, $modules, $modulelogs, $slideid)
    #
}