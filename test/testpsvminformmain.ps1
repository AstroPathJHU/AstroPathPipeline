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
Class testpsvminformmain : testtools {
    #
    [string]$processloc
    [string]$module = 'vminform'
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$class = 'vminformmain'
    #
    testpsvminformmain() : base(){
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
        $cred = Get-Credential -Message "Provide a user name (domain\username) and password"
        $currentworkerip = 'vminform37'
        $workertaskfile = '\\BKI08\e$\andrew\AstroPathPipelinePrivate\test\testpsvminform.ps1'
        #
        Write-Host '    worker task file location:' $workertaskfile
        #
        # write workertask to a .ps1 file, put name of file into $workertaskfile
        #
        psexec -i -nobanner -accepteula -u $cred.username -p $cred.getnetworkcredential().password \\$currentworkerip `
        pwsh -noprofile -noexit -executionpolicy bypass -command "$workertaskfile" `
                *>> ($this.processloc + '\testpsvminform-job.log')
        #
        Write-Host 'test run on inform finished'
        #
    }
}
#
# launch test and exit if no error found
#
[testpsvminformmain]::new() | Out-Null
exit 0