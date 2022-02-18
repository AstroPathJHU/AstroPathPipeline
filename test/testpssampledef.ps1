<# -------------------------------------------
 testpssampledef
 Benjamin Green and Andrew Jorquera - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test the sampledef object and it's paths
 -------------------------------------------#>
#
Class testpssampledef {
    #
    [string]$mpath 
    [string]$process_loc
    [string]$basepath
    [string]$module = 'shredxml'
    [string]$slideid = 'M21_1'
    #
    testpssampledef(){
        $this.launchtests()
    }
    #
    testpssampledef($module, $slideid){
        $this.module = $module
        $this.slideid = $slideid
        $this.launchtests
    }
    #
    launchtests(){
        #
        Write-Host '---------------------test ps [sampledef]---------------------'
        $this.importmodule()
        $this.testsampledefconstruction()
        $sample = sampledef $this.mpath $this.module $this.slideid
        $this.testpaths($sample)
        $sample = sampledef -mpath $this.mpath -module $this.module -batchid '8' -project '0'
        $this.testpaths($sample, '08')
        $sample = sampledef -mpath $this.mpath -module $this.module -batchid '1' -project '0'
        $this.testpaths($sample, '01')
        Write-Host '.'
        #
    }
    #
    importmodule(){
        $apmodule = $PSScriptRoot + '/../astropath'
        Import-Module $apmodule -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing'
        $this.updatebase()
    }
    #
    [void]updatebase(){
        $r = $PSScriptRoot -replace( '/', '\')
        if ($r[0] -ne '\'){
            $root = ('\\' + $env:computername+'\'+$r) -replace ":", "$"
        } else{
            $root = $r -replace ":", "$"
        }
        $this.basepath = $root + '\data'
    }
    #
    [void]testsampledefconstruction(){
        #
        Write-Host "."
        Write-Host 'test [sampledef] constructors started'
        #
        try {
            sampledef | Out-Null
        } catch {
            Throw ('[sampledef] construction with [0] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            sampledef -mpath $this.mpath -module $this.module | Out-Null
        } catch {
            Throw ('[sampledef] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            sampledef -mpath $this.mpath -module $this.module -slideid $this.slideid | Out-Null
        } catch {
            Throw ('[sampledef] construction with [3] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            sampledef -mpath $this.mpath -module $this.module -batchid '8' -project '0' | Out-Null
        } catch {
            Throw ('[sampledef] construction with [4] input(s) failed. ' + $_.Exception.Message)
        }
        #
        Write-Host 'test [sampledef] constructors finished'
        #
    }
    #
    [void]testpaths($sampledef){
        #
        Write-Host '.'
        Write-Host 'path tests started'
        #
        Write-Host '    check basepath'
        if ($sampledef.basepath -ne $this.basepath){
            Throw ('base path not defined correctly:', $sampledef.basepath, '~=', $this.basepath -join ' ')
        }
        #
        Write-Host '    check slideid'
        if ($sampledef.slideid -ne $this.slideid){
            Throw ('slideid not defined correctly:', $sampledef.slideid, '~=', $this.slideid -join ' ')
        }
        #
        Write-Host '    check scan folder'
        if ($sampledef.scan() -ne 'Scan1'){
            Throw ('Scan folder not defined correctly:', $sampledef.scan(), '~= Scan1' -join ' ')
        }
        #
        Write-Host '    check msi path'
        $usermsipath = ($sampledef.basepath, '\', $sampledef.slideid,'\im3\Scan1\MSI' -join '')
        Write-Host '    user defined MSI folder: ' $usermsipath
        Write-Host '    sampledef defined MSI folder: ' $sampledef.MSIfolder()
        if (!([regex]::Escape($sampledef.MSIfolder()) -contains [regex]::Escape($usermsipath))){
            Throw ('MSI folder not defined correctly:', $sampledef.MSIfolder(),'~=',$usermsipath -join ' ')
        }
        #
        Write-Host '    check for files in IM3 folder'
        $usermsipath += '\*im3'
        if (!(Test-Path -Path $usermsipath)) {
            Throw 'No im3 files in MSI folder'
        }
        Write-Host '    Files in IM3 folder exist'
        #
        Write-Host '    check xml path'
        #
        $userxmlpath = $this.basepath + '\' + $this.slideid + '\im3\xml'
        Write-Host '    user defined xmlpath: ' $userxmlpath
        Write-Host '    sampleddef defined XML folder: ' $sampledef.xmlfolder()
        if (!([regex]::Escape($sampledef.xmlfolder()) -contains [regex]::Escape($userxmlpath))){
            Throw ('XML folder not correct:', $sampledef.xmlfolder(), '~=', $userxmlpath -join ' ')
        }
        Write-Host '    check for files in XML folder'
        $userxmlpath += '\*xml'
        if (!(Test-Path -Path $userxmlpath)) {
            Throw 'No xml files in MSI folder'
        }
        Write-Host '    Files in XML folder exist'
        #
        Write-Host 'path tests finished'
        #
    }
    #
     [void]testpaths($sampledef, $batchid){
        #
        Write-Host '.'
        Write-Host 'path tests batch started batchid:' $batchid
        #
        Write-Host '    check basepath'
        if ($sampledef.basepath -ne $this.basepath){
            Throw ('base path not defined correctly:', $sampledef.basepath, '~=', $this.basepath -join ' ')
        }
        #
        Write-Host '    check batch'
        if ($sampledef.slideid -ne $batchid){
            Throw ('slideid not defined correctly:', $sampledef.slideid, '~=', $batchid -join ' ')
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
[testpssampledef]::new() | Out-Null
exit 0
