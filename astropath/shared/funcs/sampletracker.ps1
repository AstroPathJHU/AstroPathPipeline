function sampletracker {
    param(
        [parameter()][string]$mpath,
        [parameter()][vminformqueue]$vmq,
        [parameter()][hashtable]$modules
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
    return [sampletracker]::new($mpath, $vmq, $modules)
    #
}