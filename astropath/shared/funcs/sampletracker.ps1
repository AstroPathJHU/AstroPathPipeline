function sampletracker {
    param(
        [parameter()][string]$mpath,
        [parameter()][vminformqueue]$vmq
    )
    #
    if (!($PSBoundParameters.ContainsKey('vmq'))){
        return [sampletracker]::new($mpath)
    }
    #
    return [sampletracker]::new($mpath, $vmq)
    #
}