<# ---------------------------------------------------------
For each sample directory not control extract the xml file 
formats
 ---------------------------------------------------------#>


Function ShredXMLCohort {
    param ($root)
    $sample_all = Get-ChildItem -Path $root -Directory
    #
    foreach ($sub in $sample_all) {
        $IM3_fd = $root + "\" + $sub.Name + "\im3\"
        $ii = test-path -path $IM3_fd
        #
        $XML_fd = $root + "\" + $sub.Name + "\im3\xml\"
        $ii2 = test-path -path $XML_fd
        #
        IF (!$ii -or $sub.Name -like "Control*" -or $ii2) {
            Continue
        }
        #
        $code = $PSScriptRoot + '\ConvertIM3Path.ps1'
        & $code $root $XML_fd $sub -s -xml
        #
        $sor = $XML_fd + "\" + $sub.Name
        XCOPY $sor $XML_fd /q /y /z > NULL
        Remove-Item $sor -Recurse

    }
    #
}
#
ShredXMLCohort '\\bki04\Clinical_Specimen_3'

