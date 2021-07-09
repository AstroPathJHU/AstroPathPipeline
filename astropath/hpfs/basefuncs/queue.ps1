function Queue{
    param(
        [Parameter()][string]$module = '',
        [Parameter()][PSCredential]$Credential = [PSCredential]::Empty, 
        [Parameter()][string]$mpath='\\bki04\astropath_processing'
    )
    #
    if($Credential -eq [PSCredential]::Empty){
        $Credential = Get-Credential -Message "Provide a user name (domain\username) and password"
    } # error catch on credential
    <#
    $LoadedModules = Get-Module | Select Name
    if (!$LoadedModules -like "AstroPathPipeline") {
        Import-Module -Name $PScriptRoot + '\..\..\..\AstroPathPipeline'
    }
    #>
    $q = [queue]::new($mpath, $module, $Credential)
    #
}