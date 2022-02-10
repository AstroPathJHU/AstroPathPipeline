function sampledb {
    param(
        [parameter()][string]$mpath,
        [parameter()][string]$projects
    )
    #
    if (!($PSBoundParameters.ContainsKey('mpath'))){
        return [sampledb]::new()
    }
    #
    if (!($PSBoundParameters.ContainsKey('projects'))){
        return [sampledb]::new($mpath)
    }
    #
    return [sampledb]::new($mpath, $projects)
    #
}