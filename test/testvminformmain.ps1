using module .\testtools.psm1
<# -------------------------------------------
 testvminformmain
 created by: Andrew Jorquera
 Last Edit: 03.01.2022
 --------------------------------------------
 Description
 starts the virtual machine to run the 
 vminform tests
 -------------------------------------------#>
#
Class testvminformmain : testtools {
    #
    [string]$processloc
    [string]$module = 'vminform'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    #
    testvminformmain() : base(){
        #
        $this.testruninform()
        #
    }
    #
    [void]testruninform(){
        #
        Write-Host '.'
        Write-Host 'test run on inform started'
        #
        $this.importmodule()
        #
        $cred = Get-Credential -Message "Provide a user name (domain\username) and password"
        $dis = dispatchtasks $this.module $cred $this.mpath -test
        $this.starttestjob($dis)
        #
        Write-Host 'test run on inform finished'
        #
    }
    <# --------------------------------------------
    starttestjob
    runs powershell operation on virtual
    machine to run tests
    --------------------------------------------#>
    [void]starttestjob($dis){
        #
        $creds = $dis.GetCreds()  
        $currentworkerip = 'vminform38'
        $workertaskfile = '\\BKI08\h$\andrew\AstroPathPipelinePrivate\test\testvminform.ps1'
        #
        Write-Host '    worker task file location:' $workertaskfile
        #
        # write workertask to a .ps1 file, put name of file into $workertaskfile
        #
        psexec -i -nobanner -accepteula -u $creds[0] -p $creds[1] \\$currentworkerip `
        pwsh -noprofile -noexit -executionpolicy bypass -command "$workertaskfile" `
                *>> ($this.processloc + '\testvminform-job.log')
        #
    }
}
#
# launch test and exit if no error found
#
[testvminformmain]::new() | Out-Null
exit 0