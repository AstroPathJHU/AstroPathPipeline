function queue {
    param(
        [parameter()][string]$mpath,
        [parameter()][string]$module
    )
    return [queue]::new($mpath, $module)
}