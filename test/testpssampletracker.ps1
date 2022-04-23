﻿using module .\testtools.psm1
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
        $this.testsampletrackerconstructors()
        $sampletracker = sampletracker -mpath $this.mpath -vmq (vminformqueue $this.mpath)
        $this.testchecklog($sampletracker)
        $this.testmodules($sampletracker)
        #
        Write-Host '.'
        Write-Host 'preparing sampletracker & dir started'
        Write-Host '    sample def slide'
        $sampletracker.sampledefslide($this.slideid)
        Write-Host '    module status'
        $sampletracker.defmodulestatus()
        $sampletracker.teststatus = $true
        $this.savephenotypedata($sampletracker)
        Write-Host '    cleanup'
        $this.cleanup($sampletracker)
        Write-Host 'prepareing sampletracker & dir finished'
        #
        $this.teststatus($sampletracker)
        $this.testupdate($sampletracker, 'transfer', 'shredxml')
        $this.testupdate($sampletracker, 'shredxml', 'meanimage')
        $this.testupdate($sampletracker, 'meanimage', 'batchflatfield')
        #$this.testupdate($sampletracker, 'meanimage', 'batchmicomp')
        #$this.testupdate($sampletracker, 'batchmicomp', 'batchflatfield')
        $this.testupdate($sampletracker, 'batchflatfield', 'warpoctets')
        $this.testupdate($sampletracker, 'warpoctets', 'batchwarpkeys')
        $this.testupdate($sampletracker, 'batchwarpkeys', 'batchwarpfits')
        $this.testupdate($sampletracker, 'batchwarpfits', 'imagecorrection')
        $this.testupdate($sampletracker, 'imagecorrection', 'vminform')
        $this.testupdate($sampletracker, 'vminform', 'merge')
        $this.testupdate($sampletracker, 'merge', 'imageqa')
        $this.testupdate($sampletracker, 'imageqa', 'segmaps')
        $this.testupdate($sampletracker, 'segmaps', 'dbload')
        #
        $this.cleanup($sampletracker)
        $this.testcorrectionfile($sampletracker, $true)
        $this.returnphenotypedata($sampletracker)
        #>
        $this.testgitstatus($sampletracker)  
        #
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
            sampletracker -mpath $this.mpath -debug | Out-Null
            # $sampletracker.removewatchers()
        } catch {
            Throw ('[sampletracker] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try{
            sampletracker -mpath $this.mpath -vmq (vminformqueue $this.mpath) | Out-Null
        } catch {
            Throw ('[sampletracker] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }   
        #
        Write-Host 'test [sampletracker] constructors finished'
        #                
    }
    #
    [void]savephenotypedata($sampletracker){
        #
        write-host '    saving inform results'
        if ((test-path ($this.processloc + '\tables')) -and
            !(test-path $sampletracker.mergefolder())){
                return 
            }
        #
        $sampletracker.removedir($this.processloc + '\tables')
        $sampletracker.copy($sampletracker.mergefolder(),
            ($this.processloc + '\tables'))
            $sampletracker.removedir($this.processloc + '\Component_Tiffs')
        $sampletracker.copy($sampletracker.componentfolder(),
            ($this.processloc + '\Component_Tiffs'))
        #
    }
    #
    [void]returnphenotypedata($sampletracker){
        #
        write-host '    returning inform results'
        if (test-path ($this.processloc + '\tables')){
            $sampletracker.removedir($sampletracker.mergefolder())
            $sampletracker.copy(($this.processloc + '\tables'),
                $sampletracker.mergefolder())
        }
        #
        if (test-path ($this.processloc + '\Component_Tiffs')){
            $sampletracker.removedir($sampletracker.componentfolder())
            $sampletracker.copy(($this.processloc + '\Component_Tiffs'),
                $sampletracker.componentfolder())
            $sampletracker.removedir($this.processloc)
        }
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
        $cmodules = @('transfer', 'shredxml', 'meanimage', 'batchflatfield', 'batchmicomp', 'imagecorrection',
            'warpoctets', 'batchwarpkeys', 'batchwarpfits', 'vminform', 'merge', 'imageqa', 'segmaps', 'dbload')
        $out = Compare-Object -ReferenceObject $sampletracker.modules  -DifferenceObject $cmodules
        if ($out){
            Throw ('module lists in [sampletracker] does not match, this may indicate new modules or a typo:' + $out)
        }
        #
    }
    #
    [void]testchecklog($sampletracker){
        write-host '.'
        write-host 'test if the check log and module logs work started'
        Write-Host '    test the module logs'
        Write-Host '    module logs:' ($sampletracker.modulelogs.segmaps.('1') |
             format-table | out-string)
        #
        if (!($sampletracker.modulelogs.segmaps.('1'))){
            write-host 'no log detected works'
        }
        #
        write-host 'test if the check log and module logs work finished'
        throw 'end'
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
        $this.setstart($sampletracker, $log, $current)
        Start-Sleep -s 10
        $this.setfinish($sampletracker, $log, $current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '            '$current':' $this.getstatus($sampletracker, $current)
        Write-Host '            '$next':' $this.getstatus($sampletracker, $next)
        #
        if ($this.getstatus($sampletracker, $current) -notmatch 'READY'){
            Throw ($current + ' status not correct on finish with improper results.')
        }
        #
        if ($this.getstatus($sampletracker, $next) -notmatch 'WAITING'){
            Throw ($next + ' status not correct on finish with improper results.')
        }
        #
        Write-Host '    adding results for' $current 'and checking for normal behavior'
        #
        $this.('add' + $current + 'examples')($sampletracker)
        #
        $this.setstart($sampletracker, $log, $current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '        when started:'
        Write-Host '            '$current':' $this.getstatus($sampletracker, $current)
        Write-Host '            '$next':' $this.getstatus($sampletracker, $next) 
        #
        if ($this.getstatus($sampletracker, $current) -notmatch 'RUNNING'){
            Throw ($current + ' status not correct on running.')
        }
        #
        if ($this.getstatus($sampletracker, $next) -notmatch 'WAITING'){
            Throw ($next + ' status not correct on running.')
        }
        #
        Start-Sleep -s 10
        #
        $this.setfinish($sampletracker, $log, $current)
        #
        if ($current -match 'imageqa'){
            $this.addimageqafinished($sampletracker)
        }
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '        when finished:'
        #
        Write-Host '            '$current':' $this.getstatus($sampletracker, $current)
        Write-Host '            '$next':' $this.getstatus($sampletracker, $next) 
        #
        if ($this.getstatus($sampletracker, $current) -notmatch 'FINISHED'){
            Throw ($current + ' status not correct in intial finish')
        }
        #
        if ($this.getstatus($sampletracker, $next)  -notmatch 'READY'){
            Throw ($next + ' status not correct in intial ready.')
        }
        #
        Write-Host '    Check logs on restart'
        #
        Start-Sleep -s 10
        #
        $this.setstart($sampletracker, $log, $current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '        when started:'
        #
        Write-Host '            '$current':' $this.getstatus($sampletracker, $current)
        Write-Host '            '$next':' $this.getstatus($sampletracker, $next) 
        #
        if ($this.getstatus($sampletracker, $current) -notmatch 'RUNNING'){
            Throw ($current + ' status not correct on running.')
        }
        #
        if ($this.getstatus($sampletracker, $next)  -notmatch 'WAITING'){
            Throw ($next + ' status not correct on running.')
        }
        #
        Write-Host '    Check logs on error' 
        #
        Start-Sleep -s 10
        #
        $log.error('blah de blah de blah')
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '        when error added:'
        #
        Write-Host '            '$current':' $this.getstatus($sampletracker, $current)
        Write-Host '            '$next':' $this.getstatus($sampletracker, $next) 
        #
        if ($this.getstatus($sampletracker, $current) -notmatch 'RUNNING'){
            Throw ($current + ' status not correct on error. ')
        }
        #
        if ($this.getstatus($sampletracker, $next)  -notmatch 'WAITING'){
            Throw ($next + ' status not correct on error. ')
        }
        #
        start-sleep -s 10
        #
        $this.setfinish($sampletracker, $log, $current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '        when finished:'
        #
        Write-Host '            '$current':' $this.getstatus($sampletracker, $current)
        Write-Host '            '$next':' $this.getstatus($sampletracker, $next) 
        #
        if ($this.getstatus($sampletracker, $current) -notmatch 'ERROR'){
            Throw ($current + ' status not correct on finish w error.')
        }
        #
        if ($this.getstatus($sampletracker, $next)  -notmatch 'WAITING'){
            Throw ($next + ' status not correct on finish w error. ')
        }
        #
        Write-Host '    check clean run'
        #
        $this.setstart($sampletracker, $log, $current)
        start-sleep -s 10
        $this.setfinish($sampletracker, $log, $current)
        #
        $sampletracker.getlogstatus($current)
        $sampletracker.getlogstatus($next)
        #
        Write-Host '        when finished:'
        #
        Write-Host '            '$current':' $this.getstatus($sampletracker, $current)
        Write-Host '            '$next':' $this.getstatus($sampletracker, $next) 
        #
        if ($this.getstatus($sampletracker, $current) -notmatch 'FINISHED'){
            Throw ($current + ' status not correct on final finish. ')
        }
        #
        if ($this.getstatus($sampletracker, $next)  -notmatch 'READY'){
            Throw ($next + ' status not correct on final finish. ')
        }
        #
        Write-Host 'test' $current 'status update finished'
        #
    }
    #
    [void]setstart($sampletracker, $log, $module){
        #
        if ($module -match 'vminform'){
            $log.val = $this.task
            $sampletracker.vmq.openmainqueue($false)
        }
        #
        if ($module -match 'vminform'){
            $sampletracker.vmq.openmainqueue($false)
            $sampletracker.antibodies | ForEach-Object{
                $log.val.antibody = $_
                $log.start($module)
            }
        } else {
            $log.start($module)
        }
        #
    }
    #
    [void]setfinish($sampletracker, $log, $module){
        #
        if ($module -match 'vminform'){
            $log.val = $this.task
            $sampletracker.vmq.openmainqueue($false)
        }
        #
        if ($module -match 'vminform'){
            $sampletracker.vmq.openmainqueue($false)
            $sampletracker.antibodies | ForEach-Object{
                $log.val.antibody = $_
                $log.finish($module)
            }
        } else {
            $log.finish($module)
        }
        #
    }
    #
    [string]getstatus($sampletracker, $module){
        #
        if ($module -contains 'vminform'){
            $status = ''
            foreach ($abx in $sampletracker.antibodies){
                $status = $sampletracker.moduleinfo.($module).($abx).status
                if ($status -ne 'FINISHED'){
                    break
                }
            }
        } else {
            $status = $sampletracker.moduleinfo[$module].status
        }
        #
        return $status
        #
    }
    #
    [void]cleanup($sampletracker){
        #
        Write-Host '.'
        Write-Host 'clearing logs started'
        $sampletracker.removefile($this.mpath + '\across_project_queues\vminform-queue.csv')
        $sampletracker.copy(($this.mpath + '\vminform-queue.csv'),
         ($this.mpath + '\across_project_queues'))
        $sampletracker.removefile(
            $sampletracker.vmq.localqueuefile.($this.project))
        $sampletracker.removedir($sampletracker.informfolder())
        $sampletracker.removefile($sampletracker.slidelogbase('shredxml'))
        $sampletracker.removefile($sampletracker.slidelogbase('meanimage'))
        $sampletracker.removefile($sampletracker.slidelogbase('vminform'))
        $sampletracker.removefile($sampletracker.mainlogbase('vminform'))
        $this.removemeanimageexamples($sampletracker)
        $sampletracker.removefile($sampletracker.mainlogbase('batchmicomp'))
        $sampletracker.removefile($sampletracker.mainlogbase('batchflatfield'))
        $sampletracker.removefile($sampletracker.slidelogbase('warpoctets'))
        $this.removewarpoctetsexamples($sampletracker)
        $sampletracker.removefile($sampletracker.mainlogbase('batchwarpfits'))
        $this.removebatchwarpfitsexamples($sampletracker)
        $sampletracker.removefile($sampletracker.mainlogbase('batchwarpkeys'))
        $this.removebatchwarpkeysexamples($sampletracker)
        $sampletracker.removedir($sampletracker.basepath +'\warping')
        $sampletracker.removedir($sampletracker.warpoctetsfolder())
        $sampletracker.removedir($sampletracker.flatwim3folder())
        $sampletracker.removedir($sampletracker.flatwfolder())
        $sampletracker.removefile($sampletracker.basepath + '\upkeep_and_progress\imageqa_upkeep.csv')
        $this.removebatchflatfieldexamples($sampletracker)
        #
        Write-Host 'clearing logs finished'
        #
    }
    #
    [void]removetransferexamples($sampletracker){
        #
        $p = $sampletracker.im3folder()
        $p2 = $p + '-copy'
        #
        $sampletracker.copy($p, $p2, '*')
        $sampletracker.removedir($p)
        #
    }
    #
    [void]addtransferexamples($sampletracker){
        #
        $p = $sampletracker.im3folder()
        $p2 = $p + '-copy'
        #
        $sampletracker.copy($p2, $p, '*')
        $sampletracker.removedir($p2)
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
        $this.removetestfiles($sampletracker,
            $sampletracker.meanimagefolder(), '-mask_stack.bin')
        #
    }
    #
    [void]addmeanimageexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.meanimagefolder(), $sampletracker.meanimagereqfiles)
        $this.addtestfiles($sampletracker,
            $sampletracker.meanimagefolder(), '-mask_stack.bin')
        #
    }
    #
    [void]removebatchmicompexamples($sampletracker){
        #
        $p = $this.aptempfullname($sampletracker, 'micomp')
        $p2 = $sampletracker.micomp_fullfile($this.mpath)
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
        $p2 = $sampletracker.micomp_fullfile($this.mpath)
        #
        $micomp_data = $sampletracker.importmicomp($sampletracker.mpath, $false)
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
        $p3 = $sampletracker.mpath + '\flatfield\flatfield_'+$this.pybatchflatfieldtest+'.bin'
        $sampletracker.removefile($p3)
        #
    }
    #
    [void]addbatchflatfieldexamples($sampletracker){
        #
        $this.testcorrectionfile($sampletracker, $true)
        #
    }
    # 
    [void]addcorrectionfile($sampletracker){
        #
        $p = $this.aptempfullname($sampletracker, 'corrmodels')
        $p2 = $sampletracker.corrmodels_fullfile($this.mpath)
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
            $sampletracker.warpoctetsfolder(), 
            $sampletracker.warpoctetsreqfiles)
        #
    }
    #
    [void]addwarpoctetsexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpoctetsfolder(), 
            $sampletracker.warpoctetsreqfiles)
        #
    }
    #
    [void]removebatchwarpkeysexamples($sampletracker){
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpbatchoctetsfolder(), 
            $sampletracker.batchwarpkeysreqfiles)
        #
    }
    #
    [void]addbatchwarpkeysexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpbatchoctetsfolder(),
            $sampletracker.batchwarpkeysreqfiles)
        #
    }
    #
    [void]removebatchwarpfitsexamples($sampletracker){
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpbatchfolder(),
            $sampletracker.batchwarpfitsreqfiles)
        #
    }
    #
    [void]addbatchwarpfitsexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpbatchfolder(),
            $sampletracker.batchwarpfitsreqfiles)
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
        $this.addalgorithms($sampletracker)
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.flatwfolder(),
            $sampletracker.imagecorrectionreqfiles[0], 
            $sampletracker.im3constant
        )
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.flatwfolder(),
            $sampletracker.imagecorrectionreqfiles[1], 
            $sampletracker.im3constant
        )
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.flatwim3folder(),
            $sampletracker.imagecorrectionreqfiles[2], 
            $sampletracker.im3constant
        )
        #
    }
    #
    [void]addalgorithms($sampletracker){
        #
        $sampletracker.findantibodies()
        foreach ($abx in $sampletracker.antibodies) {
            #
            $sampletracker.vmq.checkfornewtask($this.project, $this.slideid, $abx)
            $sampletracker.vmq.localqueue.($this.project) |    
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } |
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                }
            $sampletracker.vmq.writelocalqueue($this.project)
            #
            $sampletracker.vmq.coalescevminformqueues($this.project)
            #
            $sampletracker.vmq.maincsv | 
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } | 
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                    $_.ProcessingLocation = ''
                }
                #
        }
        #
        
        #
    }
    #
    [void]removevminformexamples($sampletracker){
        #
        $this.addalgorithms($sampletracker)
        $sampletracker.removedir($sampletracker.informfolder())
        #
    }
    #
    [void]addvminformexamples($sampletracker){
        #
        $this.addalgorithms($sampletracker)
        #
        foreach ($abx in $sampletracker.antibodies ) {
            $sampletracker.cantibody = $abx
            $this.addtestfiles($sampletracker, 
                $sampletracker.cantibodyfolder(),
                $sampletracker.vminformreqfiles[0], 
                $sampletracker.im3constant)
            $this.addtestfiles($sampletracker, 
                $sampletracker.cantibodyfolder(),
                $sampletracker.vminformreqfiles[1], 
                $sampletracker.im3constant)
                #
            $sampletracker.vmq.maincsv | 
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } | 
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                    $_.ProcessingLocation = 'go place'
                }
            #
        }
        $sampletracker.vmq.writemainqueue($sampletracker.vmq.mainqueuelocation())
        #
        $this.addtestfiles($sampletracker, 
                $sampletracker.componentfolder(),
                $sampletracker.vminformreqfiles[2], 
                $sampletracker.im3constant)
        #
    }
    #
    [void]removemergeexamples($sampletracker){
        #
        $sampletracker.removedir(
            $sampletracker.mergefolder()
        )
        #
    }
    #
    [void]addmergeexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.mergefolder(),
            $sampletracker.mergereqfiles[0], 
            $sampletracker.im3constant)
        $files = Get-ChildItem $sampletracker.mergefolder()
        #
        $date = get-date
        foreach ($file in $files){
            $file.LastWriteTime = $date
        }
        #
    }
    #
    [void]removeimageqaexamples($sampletracker){
        #
        $sampletracker.removefile($sampletracker.imageqa_fullpath())
        #
    }
    #
    [void]addimageqaexamples($sampletracker){}
    #
    [void]addimageqafinished($sampletracker){
        #
        $this.removeimageqaexamples($sampletracker)
        $sampletracker.findantibodies()
        $sampletracker.ImportImageQA()
        #
        # add in Xs for all antibodies of
        # the slideid
        #
        $newline = $this.slideid
        $sampletracker.antibodies | 
            ForEach-Object {
                $newline += ',X'
            }
        #
        $newline += ",`r`n"
        #
        $sampletracker.popfile($sampletracker.imageqa_fullpath(), $newline)
        #
    }
    #
    [void]removesegmapsexamples($sampletracker){
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.segmapfolder(),
            $sampletracker.segmapsreqfiles[0],
            $sampletracker.im3constant)
        #
    }
    #
    [void]addsegmapsexamples($sampletracker){
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.segmapfolder(),
            $sampletracker.segmapsreqfiles[0],
            $sampletracker.im3constant)
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
    Throw $_.Exception
}
#
exit 0
