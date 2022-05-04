using module .\testtools.psm1
 <# -------------------------------------------
 testpssampledb
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the sample db works correctly
 and defines all the stages \ dependencies
 properly.
 -------------------------------------------#>
#
 Class testpssampledb : testtools {
    #
    [string]$class = 'sampletracker'
    [string]$module = 'sampletracker'
    #
    testpssampledb() : base(){
        $this.launchtests()
    }
    #
    testpssampledb($project, $slideid) : base($project, $slideid){
        $this.launchtests()
    }
    #
    [void]launchtests(){
        #
        $this.testpssampledbconstructors()
        $sampledb = sampledb -mpath $this.mpath
        #
        $this.cleanup($sampledb)
        $sampledb = sampledb -mpath $this.mpath
        $this.testpssampledbinit($sampledb)
        $this.testdefstages($sampledb)
        #$this.testdefsStagesParallel($sampledb)
        $this.testcreatewatchersmodulequeues($sampledb)
        #
        write-host '.'
        write-host 'get sample stages for test pipeline started'
        $sampledb.buildsampledb()
        $this.checkrowstatus($sampledb, 'transfer', 'FINISHED')
        $sampledb.preparesample($this.slideid)
        $this.savephenotypedata($sampledb)
        $this.removesetupvminform($sampledb)
        #
        $this.checkrowstatus($sampledb, 'transfer', 'FINISHED')
        $this.testwritemain($sampledb, 'transfer')
        $this.checkrowstatus($sampledb, 'transfer', 'FINISHED')
        $this.testaddrerun($sampledb, 'transfer')
        $this.testupdatemoduletables($sampledb, 'transfer')
        $this.setupbatchwarpkeys($sampledb)
        $this.checkrowstatus($sampledb, 'shredxml', 'FINISHED')
        write-host '.'
        $this.checkrowstatus($sampledb, 'meanimage', 'FINISHED')
        write-host '.'
        $this.checkrowstatus($sampledb, 'batchflatfield', 'FINISHED')
        write-host '.'
        $this.checkrowstatus($sampledb, 'warpoctets', 'FINISHED')
        write-host '.'
        $this.checkrowstatus($sampledb, 'batchwarpkeys', 'FINISHED')
        $this.testupdatemoduletables($sampledb, 'batchwarpkeys')
        #
        $this.testupdatemoduletablesVM($sampledb)
        $sampledb.preparesample($this.slideid)
        $this.removesetupvminform($sampledb)
        $this.returnphenotypedata($sampledb)
        $this.cleanup($sampledb)
        $this.testcorrectionfile($sampledb, $true)
        $this.testgitstatus($sampledb)
        write-host 'get sample stages for test pipeline finished'
        #
        Write-Host '.'
        #
    }
    #
    [void]testpssampledbconstructors(){
        #
        Write-Host "."
        Write-Host 'test [sampledb] constructors started'
        #
        try{
            sampledb $this.mpath -debug | Out-Null
            # $sampletracker.removewatchers()
        } catch {
            Throw ('[sampledb] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        <#
        try{
            sampledb $this.mpath $this.project | Out-Null
        } catch {
            Throw ('[sampledb] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }   
        #>
        Write-Host 'test [sampledb] constructors finished'
        #                
    }
    #
    [void]cleanup($sampledb){
        #
        write-host '.'
        write-host 'clean up started'
        $sampledb.removedir($this.mpath + '\across_project_queues')
        $sampledb.removedir($this.basepath + '\upkeep_and_progress\progress_tables')
        $sampledb.createnewdirs($this.basepath + '\logfiles')
        write-host 'clean up finished'
        #
    }
    #
    [void]testpssampledbinit($sampledb){
        #
        write-host '.'
        write-host 'test the status of the sampledb on init started'
        write-host '    module object (physical queues) keys:' ($sampledb.moduleobjs.Keys)
        write-host '    module log keys:' ($sampledb.modulelogs.Keys)
        write-host 'test the status of the sampledb on init finished'
        #
    }
    #
    [void]testdefstages($sampledb){
        #
        write-host '.'
        write-host 'test getting the sample status started'
        $sampledb.defsamplestages()
       # write-host '    slides added:' ($sampledb.Keys)
        #Write-host '    this.slide status:' ($sampledb.($this.slideid) | Out-string)
        write-host 'test getting the sample status finished'
        #
    }
    #
    [void]testdefsStagesParallel($sampledb){
        #
        write-host '.'
        write-host 'test getting the sample status in parallel started'
        $sampledb.defsampleStagesParallel()
        write-host '    slides added:' ($sampledb.sampledb.Keys)
        Write-host '    this.slide status:' ($sampledb.sampledb.('MA12') | Out-string)
        write-host 'test getting the sample status in parallel finished'
        #
    }
    #
    [void]testcreatewatchersmodulequeues($sampledb){
        #
        write-host '.'
        write-host 'test check create watcher update started'
        $sampledb.defsampleStages()
        write-host '    run create watchers for transfer module'
        $sampledb.moduleobjs.('transfer').createwatchersqueues()
        #
        if ($sampledb.moduleobjs.('transfer').newtasks){
            write-host '    new tasks:' $sampledb.moduleobjs.('transfer').newtasks
            Throw 'new tasks should be empty'
        }
        #
        write-host '    show the local queue after coalesce'
        write-host ($sampledb.moduleobjs.('transfer').localqueue.('0') | format-table | Out-String)
        if (!($sampledb.moduleobjs.('transfer').localqueue.('0'))){
            Throw 'the local queue should not be empty'
        }
        #
        write-host '    remove the watcher for the transfer module'
        $SI = $sampledb.moduleobjs.('transfer').mainqueuelocation()
        $sampledb.UnregisterEvent($SI)
        write-host '    create watchers for all module files'
        #
        $sampledb.defmodulewatchers()
        #
        $this.checknewtasks($sampledb)
        #
        write-host '    create watchers for all log files'
        #
        $sampledb.defmodulelogwatchers()
        #
        if ($sampledb.newtasks){
            write-host '    new tasks:' $sampledb.newtasks
            Throw 'new tasks should be empty'
        }
        #
        write-host '    refresh the module db'
        $sampledb.refreshmoduledb()
        #
        write-host 'test check create watcher update finished'
        #
    }   
    #
    [void]testupdatemoduletables($sampledb, $cmodule){
        #
        write-host '.'
        write-host 'test update module queues started -' $cmodule
        #
        $this.addrerun($sampledb, $cmodule)
        Write-host '        check the corresponding current step in the module table'
        #
        $moduleobj = $sampledb.moduleobjs.($cmodule)
        $moduleobj.openmainqueue($false)
        write-host '    new tasks:' $moduleobj.newtasks
        if (!$moduleobj.newtasks){
            throw 'new module tasks should exist'
        }
        #
        $slide, $moduleobj.newtasks = $moduleobj.newtasks
        #
        write-host '    slide task 1:' $slide
        if ($slide -notmatch $this.slideid){
            Throw 'slide not defined correctly'
        }
        #
        if ($moduleobj.newtasks){
            throw 'more than one task existed this is an error'
        }
        #
        write-host '    run through refresh manually'
        #
        write-host '        prepare sample'
        $sampledb.preparesample($slide)
        write-host '        update module db'
        $sampledb.refreshmoduledb($cmodule, $slide)
        $this.checkrowstatus($sampledb, $cmodule, $sampledb.status.READY)
        write-host '        coalescequeues'
        $moduleobj.writemainqueue()
        $moduleobj.coalescequeues()
        #
        $this.checkrowstatus($sampledb, $cmodule, $sampledb.status.READY)
        $this.checknewtasks($sampledb, $cmodule)
        #
        write-host '    run through refresh auto'
        $this.addrerun($sampledb, $cmodule)
        $sampledb.refreshmoduledb($cmodule)
        $this.checkrowstatus($sampledb, $cmodule, $sampledb.status.READY)
        Write-host '    edit a log for a finished message and add corresponding "finished" files'
        $this.addstartlog($sampledb, $cmodule)
        start-sleep 2
        $this.addfinishlog($sampledb, $cmodule)
        Write-host '        check the corresponding current and next steps in module table'
        #
        $project = '0'
        #
        $sampledb.refreshsampledb($cmodule, $project)
        #
        $this.checkrowstatus($sampledb, $cmodule, $sampledb.status.FINISHED)
        #
        write-host 'test update module queues finished'
        #
    }   
    #
    [void]testupdatemoduletablesVM($sampledb){
        #
        write-host '.'
        write-host 'test update module queues started - vminform'
        #
        $cmodule = 'vminform'
        #
        $this.setupvminform($sampledb)
        $sampledb.newtasks = @('M21_1')
        # 
        $project = '0'
        $sampledb.refreshsampledb($cmodule, $project)
        $this.checkrowstatus($sampledb, $cmodule, $sampledb.status.READY, 'CD8')
        #
        Write-host '    edit a log for a error message'
        $this.addstartlog($sampledb, $cmodule, 'CD8')
        Start-Sleep 2
        $this.adderrorlog($sampledb, $cmodule, 'CD8')
        Start-Sleep 2
        $this.addfinishlog($sampledb, $cmodule, 'CD8')
        $this.addproccessedalgorithms($sampledb)
        Write-host '        check the corresponding current and next steps in module table'
        $project = '0'
        $sampledb.getmodulelogs($cmodule, $project)
        if (!$sampledb.newtasks){
            Throw 'should be new tasks'
        }
        $sampledb.refreshsampledb($cmodule, $project)
        $this.checkrowstatus($sampledb, $cmodule, $sampledb.status.error, 'CD8')
        #
        write-host 'test update module queues finished - vminform'
        #
    }  
    #
    [void]testupdatemodulequeuesVM($sampledb){
        #
        write-host '.'
        write-host 'test update module queues started'
        #

        #
        write-host 'test update module queues finished'
        #
    } 
    #
    [void]testwritemain($sampledb, $cmodule){
        #
        write-host '.'
        write-host 'test that the write queue function is working correctly started'
        write-host 'module is:' $cmodule
        #
        $moduleobj = $sampledb.moduleobjs.($cmodule)
        #
        write-host '    show the lastopenmaincsv before editing'
        $this.showtable($moduleobj.lastopenmaincsv)
        #
        write-host '    get a stored table and set the lastopenmaincsv to blank'
        #
        $stable = $sampledb.getstoredtable($moduleobj.lastopenmaincsv)
        $moduleobj.lastopenmaincsv = ''
        #
        $this.showtable($stable)
        if (!$stable){
            throw 'stable table was also set to null'
        }
        #
        if ($moduleobj.lastopenmaincsv){
            throw 'last open main csv table was NOT set to null'
        }
        #
        write-host '    lastopenmaincsv successfully stored and set NULL'
        #
        write-host '    store main csv'
        $mainqueue = $sampledb.getstoredtable($moduleobj.maincsv)
        #
        write-host '    table to print'
        $this.showtable($mainqueue)
        #
        write-host '    open new queue'
        $moduleobj.openmainqueue($false)
        #
        $this.checknewtasks($sampledb, $cmodule)
        #
        $moduleobj.getnewtasksmain($stable, $moduleobj.lastopenmaincsv)
        #
        $this.checknewtasks($sampledb, $cmodule)
        #
        if (!($moduleobj.lastopenmaincsv)){
            throw 'last open main csv table was reset'
        }
        #
        write-host 'test that the write queue function is working correctly finished'
        #
    }
    #
    [void]testaddrerun($sampledb, $cmodule){
        #
        write-host '.'
        write-host 'test when a sample status is changed to rerun started'
        #
        $moduleobj = $sampledb.moduleobjs.($cmodule)
        $this.addrerun($sampledb, $cmodule)
        #
        $this.checkrowstatus($sampledb, $cmodule, 'FINISHED')
        $this.checkrowstatus($moduleobj.lastopenmaincsv, 'FINISHED')
        #
        write-host '        open table after edit and show'
        $moduleobj.openmainqueue($false)
        $this.checkrowstatus($sampledb, $cmodule, 'RERUN')
        $this.checkrowstatus($moduleobj.lastopenmaincsv, 'RERUN')
        write-host '        new tasks:' $moduleobj.newtasks
        #
        if (!$moduleobj.newtasks){
            Throw 'should be a new task at the end of test'
        }
        #
        $moduleobj.newtasks = @()
        #
        $this.setfinished($sampledb, $cmodule)
        #
        write-host 'test when a sample status is changed to rerun finished'
        #
    }
    #
    [void]setfinished($sampledb, $cmodule){
         #
         $moduleobj = $sampledb.moduleobjs.($cmodule)
         $this.resetrowFINISHED($sampledb, $cmodule)
         $moduleobj.writemainqueue()
         $this.checkrowstatus($sampledb, $cmodule, 'FINISHED')
         $this.checkrowstatus($moduleobj.lastopenmaincsv, 'FINISHED')
         $moduleobj.newtasks = @()
         $this.checknewtasks($sampledb, $cmodule)
         #
    }
    #
    [void]addrerun($sampledb, $cmodule){
        #
        $moduleobj = $sampledb.moduleobjs.($cmodule)
        #
        Write-host '    edit a module queue for a rerun'
        $status = $this.getrowstatus($moduleobj.maincsv)
        write-host '        table before edit status:' $status
        #
        if ($status -match 'RERUN'){
            Throw 'status cannot start as RERUN'
        }
        #
        $row = $moduleobj.maincsv |
            Where-Object { $_.slideid -match $this.slideid}
        $row.status = 'RERUN'
        #
        write-host '        table after edit:' $row.status
        $this.checkrowstatus($sampledb, $cmodule, 'RERUN')
        #
        write-host '        write table after edit'
        $moduleobj.writemainqueue()
        $this.checknewtasks($sampledb, $cmodule)
        $this.checknewtasks($sampledb)
        #
        write-host '            show table after write'
        $this.checkrowstatus($sampledb, $cmodule, 'RERUN')
        $this.checkrowstatus($moduleobj.lastopenmaincsv, 'RERUN')
        $this.('resetrow'+ $status)($sampledb, $cmodule)
        #
    }
     #
     [void]checkrowstatus($table, $status){
        $row = $table|
            Where-Object { $_.slideid -match $this.slideid}
        if ($row.status -notmatch $status){
            throw ('row status in queue not match ' + $status)
        }
        #
    }
    #
    [void]checkrowstatus($sampledb, $cmodule, $status){
        $moduleobj = $sampledb.moduleobjs.($cmodule)
        $row = $moduleobj.maincsv |
            Where-Object { $_.slideid -match $this.slideid}
        if ($row.status -notmatch $status){
            throw ('row status in main queue not match ' +
                $status + '. status is: ' + $row.status)
        }
        #
    }
    #
    [void]checkrowstatus($sampledb, $cmodule, $status, $antibody){
        $moduleobj = $sampledb.moduleobjs.($cmodule)
        $row = $moduleobj.localqueue.($this.project) |
            Where-Object { $_.slideid -match $this.slideid}
        if ($row.($antibody + '_Status') -notmatch $status){
            throw ('row status in main queue not match ' +
                $status + '. status is: ' + $row.($antibody + '_Status'))
        }
        #
    }
    #
    [string]getrowstatus($table){
        $row = $table|
            Where-Object { $_.slideid -match $this.slideid}
        return ($row.status)
    }
    #
    [void]resetrowFINISHED($sampledb, $cmodule){
        #
        $moduleobj = $sampledb.moduleobjs.($cmodule)
        #
        write-host '        edit table back to FINISHED and show'
        write-host '            akin to resetting before the RERUN write'
        #
        $row = $moduleobj.maincsv |
            Where-Object { $_.slideid -match $this.slideid}
        $row.status = 'FINISHED'
        $row = $moduleobj.lastopenmaincsv |
            Where-Object { $_.slideid -match $this.slideid}
        $row.status = 'FINISHED'
    }
    #
    [void]resetrowREADY($sampledb, $cmodule){
        #
        $moduleobj = $sampledb.moduleobjs.($cmodule)
        #
        write-host '        edit table back to READY and show'
        write-host '            akin to resetting before the RERUN write'
        #
        $row = $moduleobj.maincsv |
            Where-Object { $_.slideid -match $this.slideid}
        $row.status = 'READY'
        $row = $moduleobj.lastopenmaincsv |
            Where-Object { $_.slideid -match $this.slideid}
        $row.status = 'READY'
    }
    #
    [void]checknewtasks($sampledb, $cmodule){
        #
        if ($sampledb.moduleobjs.($cmodule).newtasks){
            write-host '    new tasks:' $sampledb.moduleobjs.($cmodule).newtasks
            Throw 'module new tasks should be empty'
        }
    }
    #
    [void]checknewtasks($sampledb){
        #
        if ($sampledb.newtasks){
            write-host '    new tasks:' $sampledb.newtasks
            Throw 'sampledb new tasks should be empty'
        }
    }
    #
    [void]addfinishlog($sampledb, $cmodule){
        #
        if ($cmodule -match 'vminform'){
            return
        }
        #   
        if ($cmodule -match 'batch') {
            $mess = $this.project, $this.cohort,
                $this.batchid.PadLeft(2,'0'), ('FINISH:' + $cmodule), (Get-Date) -join ';'
        } else {
            $mess = $this.project, $this.cohort,
                $this.slideid, ('FINISH:' + $cmodule), (Get-Date) -join ';'
        }
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    #
    [void]addstartlog($sampledb, $cmodule){
        #
        if ($cmodule -match 'vminform'){
            return
        }
        #
        if ($cmodule -match 'batch') {
            $mess = $this.project, $this.cohort,
                $this.batchid.PadLeft(2,'0'), ('START:' + $cmodule), (Get-Date) -join ';'
        } else {
            $mess = $this.project, $this.cohort,
                $this.slideid, ('START:' + $cmodule), (Get-Date) -join ';'
        }
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    #
    [void]adderrorlog($sampledb, $cmodule, $antibody){
        #
        $mess = $this.project, $this.cohort,
            $this.slideid, ('ERROR:' + $cmodule + ' - Antibody: ' + $antibody + ' - Algorithm:'), (Get-Date) -join ';'
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    #
    [void]addfinishlog($sampledb, $cmodule, $antibody){
        #
        $mess = $this.project, $this.cohort,
            $this.slideid, ('FINISH:' + $cmodule + ' - Antibody: ' + $antibody + ' - Algorithm:'), (Get-Date) -join ';'
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    #
    [void]addstartlog($sampledb, $cmodule, $antibody){
        #
        $mess = $this.project, $this.cohort,
            $this.slideid, ('START:' + $cmodule + ' - Antibody: ' + $antibody + ' - Algorithm:'), (Get-Date) -join ';'
        $logfile = $this.basepath + '\logfiles\' + $cmodule + '.log'
        #
        $mess += "`r`n"    
        $sampledb.popfile($logfile, $mess)
        #
    }
    [void]setupbatchwarpkeys($sampledb){
        #
        Write-host '    setting up batcharpkeys dependencies for the environment'
        #   
        $sampledb.modules | ForEach-Object {
            $this.addstartlog($sampledb, $_)
            Start-Sleep 2
            $this.addfinishlog($sampledb, $_)
        }
        #
        $sampledb.preparesample($this.slideid)
        $sampletracker = $sampledb
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.meanimagefolder(), $sampletracker.meanimagereqfiles)
        $this.addtestfiles($sampletracker,
            $sampletracker.meanimagefolder(), '-mask_stack.bin')
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
        $this.testcorrectionfile($sampletracker, $true)
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpoctetsfolder(), 
            $sampletracker.warpoctetsreqfiles)
        #
        $this.addtestfiles($sampletracker, 
            $sampletracker.warpbatchoctetsfolder(),
            $sampletracker.batchwarpkeysreqfiles)
        <#
        $sampledb.getmodulelogs($false)
        write-host 'newtasks:'
        write-host $sampledb.newtasks 
        #
        $sampledb.preparesample($this.slideid)
        #>
        $sampledb.getmodulelogs($false)
        $sampledb.refreshsampledb('batchwarpkeys', '0')
        #
    }
    #
    [void]setupvminform($sampledb){
        #
        Write-host '    setting up inform dependencies for the environment'
        #
        $sampledb.preparesample($this.slideid)
        $sampletracker = $sampledb
        $sampletracker.teststatus = $true
        #
        $this.addtestfiles($sampletracker, 
        $sampletracker.warpbatchfolder(),
        $sampletracker.batchwarpfitsreqfiles)
        #
        $this.addalgorithms($sampledb)
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
        $sampledb.getmodulelogs($false)
        $sampledb.preparesample($this.slideid)
        #
    }
    #
    [void]removesetupvminform($sampledb){
        #
        Write-host '    removing inform dependencies for the environment'
        #
        $sampledb.preparesample($this.slideid)
        $sampletracker = $sampledb
        $sampletracker.teststatus = $true
        #
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
        #
        $this.removetestfiles($sampletracker,
            $sampletracker.meanimagefolder(), $sampletracker.meanimagereqfiles)
        $this.removetestfiles($sampletracker,
            $sampletracker.meanimagefolder(), '-mask_stack.bin')
        #
        $sampletracker.removefile($sampletracker.mainlogbase('batchmicomp'))
        $sampletracker.removefile($sampletracker.mainlogbase('batchflatfield'))
        $sampletracker.removefile($sampletracker.slidelogbase('warpoctets'))
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpoctetsfolder(), 
            $sampletracker.warpoctetsreqfiles)
        #
        $sampletracker.removefile($sampletracker.mainlogbase('batchwarpfits'))
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpbatchfolder(),
            $sampletracker.batchwarpfitsreqfiles)
        #
        $sampletracker.removefile($sampletracker.mainlogbase('batchwarpkeys'))
        #
        $this.removetestfiles($sampletracker, 
            $sampletracker.warpbatchoctetsfolder(), 
            $sampletracker.batchwarpkeysreqfiles)
        #
        $sampletracker.removedir($sampletracker.basepath +'\warping')
        $sampletracker.removedir($sampletracker.warpoctetsfolder())
        $sampletracker.removedir($sampletracker.flatwim3folder())
        $sampletracker.removedir($sampletracker.flatwfolder())
        $sampletracker.removefile($sampletracker.basepath + '\upkeep_and_progress\imageqa_upkeep.csv')
        #
        #
        $p = $this.aptempfullname($sampletracker, 'corrmodels')
        $p2 = $sampletracker.corrmodels_fullfile($this.mpath)
        #
        $sampletracker.removefile($p2)
        $data = $sampletracker.opencsvfile($p)
        $data | Export-CSV $p2  -NoTypeInformation
        #
        $p3 = $sampletracker.mpath + '\flatfield\flatfield_'+$this.pybatchflatfieldtest+'.bin'
        $sampletracker.removefile($p3)
        #
        #
    }
    #
    [void]addalgorithms($sampledb){
        #
        $sampledb.findantibodies()
        foreach ($abx in $sampledb.antibodies) {
            #
            $sampledb.vmq.checkfornewtask($this.project, $this.slideid, $abx)
            $sampledb.vmq.localqueue.($this.project) |    
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } |
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                }
            $sampledb.vmq.writelocalqueue($this.project)
            #
            $sampledb.vmq.coalescevminformqueues($this.project)
            #
            $sampledb.vmq.maincsv | 
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
    }
    #
    [void]addproccessedalgorithms($sampledb){
        #
        $sampledb.findantibodies()
        foreach ($abx in $sampledb.antibodies) {
            #
            $sampledb.vmq.checkfornewtask($this.project, $this.slideid, $abx)
            $sampledb.vmq.localqueue.($this.project) |    
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } |
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                }
            $sampledb.vmq.writelocalqueue($this.project)
            #
            $sampledb.vmq.coalescevminformqueues($this.project)
            #
            $sampledb.vmq.maincsv | 
                Where-Object {
                    $_.slideid -match $this.slideid -and 
                    $_.Antibody -match $abx   
                } | 
                Foreach-object {
                    $_.algorithm = 'blah.ifr'
                    $_.ProcessingLocation = 'Processing: bki##'
                }
                #
        }
        #
        $sampledb.vmq.writemainqueue($this.project)
        #
    }
}
#
# launch test and exit if no error found
#
try{
    [testpssampledb]::new() | Out-Null
} catch {
    Throw $_.Exception
}
#
exit 0
