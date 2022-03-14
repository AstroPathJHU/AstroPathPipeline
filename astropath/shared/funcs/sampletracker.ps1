function sampletracker {
    param(
        [parameter()][string]$mpath,
        [parameter()][string]$slideid
    )
    #
    if (!($PSBoundParameters.ContainsKey('project'))){
        return [sampletracker]::new($mpath)
    }
    #
    # return [sampletracker]::new($mpath, $module, $slideid, $project)
    #
}