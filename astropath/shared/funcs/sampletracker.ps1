function sampletracker {
    param(
        [parameter()][string]$mpath,
        [parameter()][string]$slideid,
        [parameter()][string]$project
    )
    #
    if (!($PSBoundParameters.ContainsKey('project'))){
        return [sampletracker]::new($mpath, $slideid)
    }
    #
    # return [sampletracker]::new($mpath, $module, $slideid, $project)
    #
}