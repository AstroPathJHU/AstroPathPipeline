using module .\testtools.psm1
 <# -------------------------------------------
 testsampletracker
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the sample tracker works correctly
 and defines all the stages \ dependencies
 properly.
 -------------------------------------------#>
#
 Class testpssampletracker : testtools {
    #
    [string]$class = 'sampletracker'
    [string]$module = ''
    #
    testpssampletracker() : base(){
        $this.launchtests()
    }
    #
    testpssampletracker($project, $slideid) : base($project, $slideid){
        $this.launchtests()
    }
    #
    [void]launchtests(){
        #
        #$this.testsampletrackerconstructors()
        $sampletracker = sampletracker -mpath $this.mpath
        $this.testmodules($sampletracker)
        #
        Write-Host '.'
        Write-Host 'defining module status started'
        $sampletracker.sampledefslide($this.slideid)
        $sampletracker.defmodulestatus()
        $sampletracker.teststatus = $true
        $this.cleanup($sampletracker)
        Write-Host 'defining module status finished'
        #
        #$this.teststatus($sampletracker)
        $this.testupdate($sampletracker, 'shredxml', 'meanimage')
        $this.testupdate($sampletracker, 'meanimage', 'batchflatfield')
        #$this.testupdate($sampletracker, 'meanimage', 'batchmicomp')
        #$this.testupdate($sampletracker, 'batchmicomp', 'batchflatfield')
        $this.testupdate($sampletracker, 'batchflatfield', 'warpoctets')
        $this.testupdate($sampletracker, 'warpoctets', 'batchwarpkeys')
        $this.testupdate($sampletracker, 'batchwarpkeys', 'batchwarpfits')
        $this.testupdate($sampletracker, 'batchwarpfits', 'imagecorrection')
        #$this.testupdate($sampletracker, 'imagecorrection', 'vminform')
        #$this.testupdate($sampletracker, 'vminform', 'segmaps')
        $this.cleanup($sampletracker)
        $this.addbatchflatfieldexamples($sampletracker)
        Write-Host '.'
        #
    }
    #
    [void]testsampletrackerconstructors(){
        #
        Write-Host "."
        Write-Host 'test [sampletracker] constructors started'
        #
        try{
            sampletracker -mpath $this.mpath | Out-Null
            # $sampletracker.removewatchers()
        } catch {
            Throw ('[sampletracker] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        #
        Write-Host 'test [sampletracker] constructors finished'
        #                
    }
    #
    [void]testmodules($sampletracker){
        #
        Write-Host "."
        Write-Host "Get module names"
        try {
            $sampletracker.getmodulenames()
        } catch {
            Throw ('[sampletracker].getmodulenames() failed: ' + $_.Exception.Message)
        }
        #
        Write-Host '    Modules:' $sampletracker.modules 
        #
        $cmodules = @('batchflatfield','batchmicomp','imagecorrection','meanimage','mergeloop',`
            'segmaps','shredxml','transfer','vminform','warpoctets','batchwarpkeys','batchwarpfits')
        $out = Compare-Object -ReferenceObject $sampletracker.modules  -DifferenceObject $cmodules
        if ($out){
            Throw ('module lists in [sampletracker] does not match, this may indicate new modules or a typo:' + $out)
        }
        #
    }
    #
    [void]teststatus($sampletracker){
        #
        Write-Host "."
        Write-Host 'test status init on' $this.slideid 'started'
        #
        try {
            $sampletracker.defmodulestatus()
        } catch {
            Throw ('[sampletracker].defmodulestatus() failed: ' + $_.Exception.Message)
        }
        #
        Write-Host ($sampletracker.moduleinfo | Out-String)
        $sampletracker.moduleinfo.keys | ForEach-Object {
            Write-Host $_ 'module info'
            Write-Host ($sampletracker.moduleinfo[$_] | Out-String)
        }
        #
        if ($sampletracker.moduleinfo.transfer.status -notmatch 'FINISHED'){
            Throw ('transfer init status not correct. Status is: ' + 
                $sampletracker.moduleinfo.transfer.status)
        }
        #
        if ($sampletracker.moduleinfo.shredxml.status -notmatch 'READY'){
            Throw ('shredxml init status not correct. Status is: ' + 
                $sampletracker.moduleinfo.shredxml.status)
        }
        #
        if ($sampletracker.moduleinfo.meanimage.status -notmatch 'WAITING'){
            Throw ('meanimage init status not correct. Status is: ' + 
                $sampletracker.moduleinfo.meanimage.status)
        }
        #
        if ($sampletracker.moduleinfo.batchmicomp.status -notmatch 'WAITING'){
            Throw ('batchmicomp init status not correct. Status is: ' + 
                $sampletracker.moduleinfo.batchmicomp.status)
        }
        #
        if ($sampletracker.moduleinfo.warpoctets.status -notmatch 'WAITING'){
            Throw ('warpoctets init status not correct. Status is: ' + 
                $sampletracker.moduleinfo.warpoctets.status)
        }
        #
        if ($sampletracker.moduleinfo.batchflatfield.status -notmatch 'WAITING'){
            Throw ('batchflatfield init status not correct. Status is: ' + 
                $sampletracker.moduleinfo.batchflatfield.status)
        }
        #
        Write-Host 'test status init on' $this.slideid 'finished'
    }
    #
    [void]testupdate($sampletracker, $current, $next){
        #
        Write-Host "."
        Write-Host 'test' $current 'status update started'
        Write-Host '    test' $current 'status with updated log but no results'
        #
        $this.('remove' + $current + 'examples')($sampletracker)
        #
        if ($current -match 'batch'){
            $log = logger -mpath $this.mpath -module $current -batchid $sampletracker.batchid -project $sampletracker.project
        } else {
            $log = logger -mpath $this.mpath -module $current -slideid $sampletracker.slideid
        }
        #
        $log.start($current)
        Start-Sleep -s 10
        $log.finish($current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '            '$current':' $sampletracker.moduleinfo[$current].status
        Write-Host '            '$next':' $sampletracker.moduleinfo[$next].status
        #
        if ($sampletracker.moduleinfo[$current].status -notmatch 'READY'){
            Throw ($current + ' status not correct on finish with improper results.')
        }
        #
        if ($sampletracker.moduleinfo[$next].status -notmatch 'WAITING'){
            Throw ($next + ' status not correct on finish with improper results.')
        }
        #
        Write-Host '    adding results for' $current 'and checking for normal behavior'
        #
        $this.('add' + $current + 'examples')($sampletracker)
        #
        $log.start($current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '        when started:'
        Write-Host '            '$current':' $sampletracker.moduleinfo[$current].status
        Write-Host '            '$next':' $sampletracker.moduleinfo[$next].status
        #
        if ($sampletracker.moduleinfo[$current].status -notmatch 'RUNNING'){
            Throw ($current + ' status not correct on running.')
        }
        #
        if ($sampletracker.moduleinfo[$next].status -notmatch 'WAITING'){
            Throw ($next + ' status not correct on running.')
        }
        #
        Start-Sleep -s 10
        #
        $log.finish($current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '        when finished:'
        #
        Write-Host '            '$current':' $sampletracker.moduleinfo[$current].status
        Write-Host '            '$next':' $sampletracker.moduleinfo[$next].status
        #
        if ($sampletracker.moduleinfo[$current].status -notmatch 'FINISHED'){
            Throw ($current + ' status not correct in intial finish')
        }
        #
        if ($sampletracker.moduleinfo[$next].status -notmatch 'READY'){
            Throw ($next + ' status not correct in intial ready.')
        }
        #
        Write-Host '    Check logs on restart'
        #
        Start-Sleep -s 10
        #
        $log.start($current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus( $next)
        #
        Write-Host '        when started:'
        #
        Write-Host '            '$current':' $sampletracker.moduleinfo[$current].status
        Write-Host '            '$next':' $sampletracker.moduleinfo[$next].status
        #
        if ($sampletracker.moduleinfo[$current].status -notmatch 'RUNNING'){
            Throw ($current + ' status not correct on running.')
        }
        #
        if ($sampletracker.moduleinfo[$next].status -notmatch 'WAITING'){
            Throw ($next + ' status not correct on running.')
        }
        Write-Host '    Check logs on error' 
        #
        Start-Sleep -s 10
        #
        $log.error('blah de blah de blah')
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus( $next)
        #
        Write-Host '        when error added:'
        #
        Write-Host '            '$current':' $sampletracker.moduleinfo[$current].status
        Write-Host '            '$next':' $sampletracker.moduleinfo[$next].status
        #
        if ($sampletracker.moduleinfo[$current].status -notmatch 'RUNNING'){
            Throw ($current + ' status not correct on error. ')
        }
        #
        if ($sampletracker.moduleinfo[$next].status -notmatch 'WAITING'){
            Throw ($next + ' status not correct on error. ')
        }
        #
        start-sleep -s 10
        #
        $log.finish($current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus( $next)
        #
        Write-Host '        when finished:'
        #
        Write-Host '            '$current':' $sampletracker.moduleinfo[$current].status
        Write-Host '            '$next':' $sampletracker.moduleinfo[$next].status
        #
        if ($sampletracker.moduleinfo[$current].status -notmatch 'ERROR'){
            Throw ($current + ' status not correct on finish w error.')
        }
        #
        if ($sampletracker.moduleinfo[$next].status -notmatch 'WAITING'){
            Throw ($next + ' status not correct on finish w error. ')
        }
        #
        Write-Host '    check clean run'
        #
        $log.start($current)
        start-sleep -s 10
        $log.finish($current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '        when finished:'
        #
        Write-Host '            '$current':' $sampletracker.moduleinfo[$current].status
        Write-Host '            '$next':' $sampletracker.moduleinfo[$next].status
        #
        if ($sampletracker.moduleinfo[$current].status -notmatch 'FINISHED'){
            Throw ($current + ' status not correct on final finish. ')
        }
        #
        if ($sampletracker.moduleinfo[$next].status -notmatch 'READY'){
            Throw ($next + ' status not correct on final finish. ')
        }
        #
        Write-Host 'test' $current 'status update finished'
        #
    }
    #
    [void]cleanup($sampletracker){
        #
        Write-Host '.'
        Write-Host 'clearing logs started'
        $sampletracker.removefile($sampletracker.slidelogbase('shredxml'))
        $sampletracker.removefile($sampletracker.slidelogbase('meanimage'))
        $sampletracker.removefile($sampletracker.mainlogbase('batchmicomp'))
        $sampletracker.removefile($sampletracker.mainlogbase('batchflatfield'))
        $sampletracker.removefile($sampletracker.mainlogbase('batchwarpfits'))
        $sampletracker.removefile($sampletracker.mainlogbase('batchwarpkeys'))
        $sampletracker.removefile($sampletracker.slidelogbase('warpoctets'))
        Write-Host 'clearing logs finished'
        #
    }
    #
    [void]removeshredxmlexamples($sampletracker){
        #
        $p = $sampletracker.xmlfolder()
        $p2 = $p + '-copy'
        #
        $sampletracker.copy($p, $p2, '*')
        $sampletracker.removedir($p)
        #
    }
    #
    [void]addshredxmlexamples($sampletracker){
        #
        $p = $sampletracker.xmlfolder()
        $p2 = $p + '-copy'
        #
        $sampletracker.copy($p2, $p, '*')
        $sampletracker.removedir($p2)
        #
    }
    #
    [void]removemeanimageexamples($sampletracker){
        #
        $this.removetestfiles($sampletracker,
            $sampletracker.meanimagefolder(), $sampletracker.meanimagereqfiles)
        #
    }
    #
    [void]addmeanimageexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.meanimagefolder(), $sampletracker.meanimagereqfiles)
        #
    }
    #
    [void]removebatchmicompexamples($sampletracker){
        #
        $p = $this.aptempfullname($sampletracker, 'micomp')
        $p2 = $sampletracker.micomp_fullfile
        #
        $sampletracker.removefile($p2)
        $data = $sampletracker.opencsvfile($p)
        $data | Export-CSV $p2 -NoTypeInformation
        #
        $this.removebatchflatfieldexamples($sampletracker)
        #
    }
    #
    [void]addbatchmicompexamples($sampletracker){
        #
        $p2 = $sampletracker.micomp_fullfile
        #
        $micomp_data = $sampletracker.importmicomp($sampletracker.mpath)
        $newobj = [PSCustomObject]@{
            root_dir_1 = $sampletracker.basepath + '\'
            slide_ID_1 = $sampletracker.slideid
            root_dir_2 = $sampletracker.basepath + '\'
            slide_ID_2 = 'blah'
            layer_n = 1
            delta_over_sigma_std_dev = .95
        }
        $micomp_data += $newobj
        #
        $micomp_data | Export-CSV $p2 -NoTypeInformation
        #
        $this.addbatchflatfieldexamples($sampletracker)
        #
    }
    #
    [void]removebatchflatfieldexamples($sampletracker){
        #
        $this.addcorrectionfile($sampletracker)
        $p3 = $sampletracker.mpath + '\flatfield\'+$this.pybatchflatfieldtest+'.bin'
        $sampletracker.removefile($p3)
        #
    }
    #
    [void]addbatchflatfieldexamples($sampletracker){
        #
        $p2 = $sampletracker.corrmodels_fullfile
        #
        $micomp_data = $sampletracker.ImportCorrectionModels($sampletracker.mpath)
        $newobj = [PSCustomObject]@{
            SlideID = $sampletracker.slideid
            Project = $sampletracker.project
            Cohort = $sampletracker.cohort
            BatchID = $sampletracker.batchid
            FlatfieldVersion = $this.pybatchflatfieldtest
            WarpingFile = 'None'
        }
        #
        $micomp_data += $newobj
        #
        $micomp_data | Export-CSV $p2 -NoTypeInformation
        $p3 = $sampletracker.mpath + '\flatfield\' + $this.pybatchflatfieldtest + '.bin'
        $sampletracker.SetFile($p3, 'blah de blah')
    }
    # 
    [void]addcorrectionfile($sampletracker){
        #
        $p = $this.aptempfullname($sampletracker, 'corrmodels')
        $p2 = $sampletracker.corrmodels_fullfile
        #
        $sampletracker.removefile($p2)
        $data = $sampletracker.opencsvfile($p)
        $data | Export-CSV $p2  -NoTypeInformation
        #
    }
    #
    [void]removewarpoctetsexamples($sampletracker){
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpoctetsfolder(), $sampletracker.warpoctetsreqfiles)
        #
    }
    #
    [void]addwarpoctetsexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpoctetsfolder(), $sampletracker.warpoctetsreqfiles)
        #
    }
    #
    [void]removebatchwarpkeysexamples($sampletracker){
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpbatchfolder(), $sampletracker.batchwarpkeysreqfiles)
        #
    }
    #
    [void]addbatchwarpkeysexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpbatchfolder(), $sampletracker.batchwarpkeysreqfiles)
        #
    }
    #
    [void]removebatchwarpfitsexamples($sampletracker){
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpbatchfolder(), $sampletracker.batchwarpfitsreqfiles)
        #
    }
    #
    [void]addbatchwarpfitsexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpbatchfolder(), $sampletracker.batchwarpfitsreqfiles)
        #
    }
    #
    [void]removeimagecorrectionexamples($sampletracker){
        #
        $sampletracker.removedir($sampletracker.flatwfolder())
        $sampletracker.removedir($sampletracker.flatwim3folder())
        #
    }
    #
    [void]addimagecorrectionexamples($sampletracker){
        #
        $sampletracker.addtestfiles($sampletracker, 
            $sampletracker.flatwfolder(), $sampletracker.imagecorrectionreqfiles[0], 
            $sampletracker.im3constant
        )
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.flatwfolder(), $sampletracker.imagecorrectionreqfiles[1], 
            $sampletracker.im3constant
        )
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.flatwim3folder(), $sampletracker.imagecorrectionreqfiles[2], 
            $sampletracker.im3constant
        )
        #
    }
    #
}
#
# launch test and exit if no error found
#
try{
    [testpssampletracker]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0
