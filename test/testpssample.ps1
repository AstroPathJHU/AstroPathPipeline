using module .\testtools.psm1
<# -------------------------------------------
 testpssample
 Benjamin Green and Andrew Jorquera - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test the sample object and it's paths
 -------------------------------------------#>
#
Class testpssample : testtools {
    #
    [string]$module = 'shredxml'
    [string]$class = 'sample'
    #
    testpssample(): base(){
        $this.launchtests()
    }
    #
    testpssample($module, $slideid) : base(){
        $this.module = $module
        $this.slideid = $slideid
        $this.launchtests()
    }
    #
    launchtests(){
        #
        $this.testsampleconstruction()
        $sample = sample $this.mpath $this.module $this.slideid
        $this.testcorrectionfile($sample, $true)
        $sample.ImportCorrectionModels($this.mpath, $true)
        $this.testpaths($sample)
        $sample = sample -mpath $this.mpath -module $this.module -batchid '8' -project '0'
        $this.testpaths($sample, '08')
        $sample = sample -mpath $this.mpath -module $this.module -batchid '1' -project '0'
        $this.testpaths($sample, '01')
        $this.testgitstatus($sample)        
        Write-Host '.'
        #
    }
    #
    [void]testsampleconstruction(){
        #
        Write-Host "."
        Write-Host 'test [sample] constructors started'
        #
        try {
            sample | Out-Null
        } catch {
            Throw ('[sample] construction with [0] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            sample -mpath $this.mpath -module $this.module | Out-Null
        } catch {
            Throw ('[sample] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            sample -mpath $this.mpath -module $this.module -slideid $this.slideid | Out-Null
        } catch {
            Throw ('[sample] construction with [3] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            sample -mpath $this.mpath -module $this.module -batchid '8' -project '0' | Out-Null
        } catch {
            Throw ('[sample] construction with [4] input(s) failed. ' + $_.Exception.Message)
        }
        #
        Write-Host 'test [sample] constructors finished'
        #
    }
    #
    [void]testpaths($sample){
        #
        Write-Host '.'
        Write-Host 'path tests started'
        #
        Write-Host '    check basepath'
        if ($sample.basepath -ne $this.basepath){
            Throw ('base path not defined correctly:', $sample.basepath, '~=', $this.basepath -join ' ')
        }
        #
        Write-Host '    check slideid'
        if ($sample.slideid -ne $this.slideid){
            Throw ('slideid not defined correctly:', $sample.slideid, '~=', $this.slideid -join ' ')
        }
        #
        Write-Host '    check scan folder'
        if ($sample.scan() -ne 'Scan1'){
            Throw ('Scan folder not defined correctly:', $sample.scan(), '~= Scan1' -join ' ')
        }
        #
        Write-Host '    check im3 path'
        $userim3path = ($sample.basepath, '\', $sample.slideid,'\im3\Scan1\MSI' -join '')
        Write-Host '    user defined im3 folder: ' $userim3path
        Write-Host '    sample defined im3 folder: ' $sample.im3folder()
        if (!([regex]::Escape($sample.im3folder()) -contains [regex]::Escape($userim3path))){
            Throw ('im3 folder not defined correctly:', $sample.im3folder(),'~=',$userim3path -join ' ')
        }
        #
        Write-Host '    check for files in IM3 folder'
        $userim3path += '\*im3'
        if (!(Test-Path -Path $userim3path)) {
            Throw 'No im3 files in im3 folder'
        }
        Write-Host '    Files in IM3 folder exist'
        #
        Write-Host '    check xml path'
        #
        $userxmlpath = $this.basepath + '\' + $this.slideid + '\im3\xml'
        Write-Host '    user defined xmlpath: ' $userxmlpath
        Write-Host '    sampleddef defined XML folder: ' $sample.xmlfolder()
        if (!([regex]::Escape($sample.xmlfolder()) -contains [regex]::Escape($userxmlpath))){
            Throw ('XML folder not correct:', $sample.xmlfolder(), '~=', $userxmlpath -join ' ')
        }
        Write-Host '    check for files in XML folder'
        $userxmlpath += '\*xml'
        if (!(Test-Path -Path $userxmlpath)) {
            Throw 'No xml files in im3 folder'
        }
        Write-Host '    Files in XML folder exist'
        #
        $ids = $sample.ImportCorrectionModels($this.mpath)
        Write-Host '    correction models file:'
        Write-Host '    ' ($ids | Format-Table | Out-String)
        #
        Write-Host '    sample py batch flatfield:' $sample.pybatchflatfield() 
        Write-Host '    test py batch flatfield:  ' $this.pybatchflatfieldtest
        if (!([regex]::escape($sample.pybatchflatfield()) `
                -contains [regex]::escape($this.pybatchflatfieldtest))){
            Throw 'py batch flatfield not detected correctly'

        }
        #
        Write-Host 'path tests finished'
        #
    }
    #
     [void]testpaths($sample, $batchid){
        #
        Write-Host '.'
        Write-Host 'path tests batch started batchid:' $batchid
        #
        Write-Host '    check basepath'
        if ($sample.basepath -ne $this.basepath){
            Throw ('base path not defined correctly:', $sample.basepath, '~=', $this.basepath -join ' ')
        }
        #
        Write-Host '    check batch'
        if ($sample.slideid -ne $batchid){
            Throw ('slideid not defined correctly:', $sample.slideid, '~=', $batchid -join ' ')
        }
        #
        Write-Host '    check py batch flatfield'
        #
        $ids = $sample.ImportCorrectionModels($this.mpath)
        Write-Host '        correction models file:'
        Write-Host '        ' ($ids | Format-Table | Out-String)
        #
        Write-Host '        sample py batch flatfield:' $sample.pybatchflatfield() 
        Write-Host '        test py batch flatfield:  ' $this.pybatchflatfieldtest
        if (!([regex]::escape($sample.pybatchflatfield()) `
                -contains [regex]::escape($this.pybatchflatfieldtest))){
            Throw 'py batch flatfield not detected correctly'

        }
        if (!(Test-Path $sample.pybatchflatfieldfullpath())){
            Throw ('flatfield file does not exist: ' + 
                $sample.pybatchflatfieldfullpath())
        }
        #
        Write-Host '    check batchslides'
        #
        $slides = $sample.importslideids($this.mpath)
        Write-Host '        slides:' ($slides | Format-Table |  Out-String)
        #
        if ($batchid[0] -match '0'){
            [string]$batchid = $batchid[1]
        }
        #
        $batch = $slides | 
            Where-Object -FilterScript {$_.BatchID -eq $batchid.trim() -and 
                $_.Project -eq $sample.project.trim()}
        #
        Write-Host '        batch:' ($batch | Format-Table | Out-String)
        Write-Host '        '($sample.batchslides | Format-Table | Out-String)
        #
        if (!$sample.batchslides){
            Throw 'no batch slides found!!'
        }
        #
        Write-Host 'path tests batch finished batchid:' $batchid
        #
    }
    #
}
#
# launch test and exit if no error found
#
try {
    [testpssample]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0

