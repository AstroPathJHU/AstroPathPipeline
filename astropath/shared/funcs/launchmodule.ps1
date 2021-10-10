function LaunchModule{
    param(
        [Parameter()][string]$slideid='',
        [Parameter()][string]$mpath='',
        [Parameter()][string]$module = '',
        [Parameter()][array]$stringin= ''
    )
    #
    $arrayin = $stringin -split '-'
    #
    if (!($PSBoundParameters.ContainsKey('slideid'))){
        $slideid = $arrayin[1]
    }
    #
    if ($module -match 'batch'){
        $m = [launchmodule]::new($mpath, $module, $slideid, $arrayin)
    } else {
        $m = [launchmodule]::new($mpath, $module, $slideid, $arrayin[0], $arrayin)
    }        
    #
}