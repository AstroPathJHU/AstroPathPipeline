function astropathworkflow{
    param(
        [Parameter()][string]$mpath='\\bki04\astropath_processing',
        [Parameter()][array]$projects,
        [Parameter()][string]$module = 'hpfs', 
        [Parameter()][array]$submodules,
        [Parameter()][PSCredential]$Credential = [PSCredential]::Empty, 
        [Parameter()][switch]$test 
    )
    #
    if($Credential -eq [PSCredential]::Empty){
        $Credential = Get-Credential -Message "Provide a user name (domain\username) and password"
    } 
    #
    if ($module -match 'hpfs'){
        #
        if (!($PSBoundParameters.ContainsKey('mpath'))){
            $aw = [astropathworkflow]::new($Credential)
        } elseif (!($PSBoundParameters.ContainsKey('projects'))){
            $aw = [astropathworkflow]::new($Credential, $mpath)
        } elseif (!($PSBoundParameters.ContainsKey('submodules'))){
            $aw = [astropathworkflow]::new($Credential, $mpath, $projects)
        }  else {
            $aw = [astropathworkflow]::new($Credential, $mpath, $projects)
        }
        #

    } else{
        #
        $aw = [astropathworkflow]::new()
        #
    }
    #
    if ($test){
        return $aw
    }
    #
}