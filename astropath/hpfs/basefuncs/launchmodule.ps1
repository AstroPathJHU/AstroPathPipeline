function LaunchModule{
    param(
        [Parameter()][string]$slideid='',
        [Parameter()][string]$mpath='',
        [Parameter()][string]$module = '',
        [Parameter()][array]$in= ''
    )
    <#
    $LoadedModules = Get-Module | Select Name
    if (!$LoadedModules -like "AstroPathPipeline") {
        Import-Module -Name $PScriptRoot + '\..\..\..\AstroPathPipeline'
    }
    #>
    $stringin = $in -replace(' '. '')
    $arrayin = $stringin -split ','
    #
    if (!($PSBoundParameters.ContainsKey('slideid'))){
        $slideid = $arrayin[1]
    }
    #
    $m = [launchmodule]::new($slideid, $mpath, $module, $arrayin)
    #
}