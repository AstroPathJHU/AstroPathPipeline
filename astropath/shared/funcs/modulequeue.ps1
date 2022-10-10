function modulequeue {
    param(
        [parameter()][string]$mpath,
        [parameter()][string]$module
    )
    return [modulequeue]::new($mpath, $module)
}