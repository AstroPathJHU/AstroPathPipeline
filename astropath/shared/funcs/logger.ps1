function logger {
    param(
        [parameter()][string]$mpath,
        [parameter()][string]$module,
        [parameter()][string]$slideid,
        [parameter()][string]$project
    )
    #
    if (!($PSBoundParameters.ContainsKey('mpath'))){
        return [mylogger]::new()
    }
    #
    if (!($PSBoundParameters.ContainsKey('module'))){
        return [mylogger]::new()
    }
    #
    if (!($PSBoundParameters.ContainsKey('slideid'))){
        return [mylogger]::new($mpath, $module)
    }
    #
    if (!($PSBoundParameters.ContainsKey('project'))){
        return [mylogger]::new($mpath, $module, $slideid)
    }
    #
    return [mylogger]::new($mpath, $module, $slideid, $project)
    #
}