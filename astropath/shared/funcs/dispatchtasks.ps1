function DispatchTasks{
    param(
        [Parameter()][string]$module = '',
        [Parameter()][PSCredential]$Credential = [PSCredential]::Empty, 
        [Parameter()][string]$mpath='\\bki04\astropath_processing',
        [Parameter()][string]$project
    )
    #
    if($Credential -eq [PSCredential]::Empty){
        $Credential = Get-Credential -Message "Provide a user name (domain\username) and password"
    } # error catch on credential
    #
    if ($module -match 'hpfs'){
        $st = [sharedtools]::new()
        $modules = @('shredxml','meanimage','batchflatfield','imagecorrection','vminform')
        $modules | foreach-object {
            $mycommand = '-command "&{Import-Module ' + $st.coderoot() + 
                '; DispatchTasks -module:'+ $_ + '-Credential:' + $Credential + ' -mpath:' + $mpath + '}"'
            Start-Process PowerShell -ArgumentList '-NoProfile', '-ExecutionPolicy Bypass', $mycommand -Verb RunAs
        }
    }
    #
    if (($PSBoundParameters.ContainsKey('project'))){
        $q = [dispatcher]::new($mpath, $module, $project, $Credential)
    } else {
        $q = [dispatcher]::new($mpath, $module, $Credential)
    }
    #
}