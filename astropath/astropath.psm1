<#
 allow all functions of the module to be .sourced into the envir,
 funcs exported below and in the 'FunctionsToExport' values in the 
 psd1 will be visible to the external user. Ideally this only includes
 functions in the public folder of the module.
 #
 adapted from general help on powershell modules:
 https://github.com/RamblingCookieMonster/PSStackExchange.git, 10.23.2020
#>
#
#Get public and private function definition files.
#
    $Private = @( Get-ChildItem -Path $PSScriptRoot\* -Include *.ps1 -Recurse -ErrorAction SilentlyContinue )
#Dot source the files
    Foreach($import in @($Private))
    {
        Try
        {
            . $import.fullname
        }
        Catch
        {
            Write-Error -Message "Failed to import function $($import.fullname): $_"
        }
    }
#
Export-ModuleMember -Function *