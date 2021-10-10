function DispatchTasks{
    param(
        [Parameter()][string]$module = '',
        [Parameter()][PSCredential]$Credential = [PSCredential]::Empty, 
        [Parameter()][string]$mpath='\\bki04\astropath_processing'
    )
    #
    if($Credential -eq [PSCredential]::Empty){
        $Credential = Get-Credential -Message "Provide a user name (domain\username) and password"
    } # error catch on credential
    $q = [dispatcher]::new($mpath, $module, $Credential)
    #
}