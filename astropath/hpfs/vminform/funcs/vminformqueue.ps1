function vminformqueue {
    param(
        [parameter()][string]$mpath,
        [parameter()][string]$project
    )
    #
    if (!($PSBoundParameters.ContainsKey('project'))){
        return [vminformqueue]::new($mpath)
    }
    #
    return [vminformqueue]::new($mpath, $project)
    #
}