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
 Class testpsvminformqueue : testtools {
    #
    [string]$class = 'vminformqueue'
    [string]$module = ''
    [string]$copypath 
    #
    testpsvminformqueue() : base(){
        $this.launchtests()
    }
    #
    testpsvminformqueue($project, $slideid) : base($project, $slideid){
        $this.launchtests()
    }
    #
    [void]launchtests(){
        #
        $this.copypath = $this.mpath + '\vminform-queue.csv'
        $vminformqueue = vminformqueue -mpath $this.mpath

        $this.testvminformqueuecontstructors()
        $vminformqueue = vminformqueue -mpath $this.mpath
        $this.resetinformqueue($vminformqueue)
        #
        $this.testopenmaininformqueue($vminformqueue)
        $this.testopenlocalqueue($vminformqueue)
        $this.testcoalescemethods($vminformqueue)
        #
        write-Host 'TEST WITH ONLY LOCAL TASKS TO START'
        $vminformqueue.removefile($vminformqueue.mainqueuelocation())
        $this.testopenmaininformqueue($vminformqueue, $true)
        $this.testopenlocalqueue($vminformqueue, $true)
        $this.testcoalescemethods($vminformqueue, $true)
        $this.resetinformqueue($vminformqueue)
        $this.testfullcoalesce($vminformqueue)
        $this.testaddinformtask($vminformqueue)
        $this.resetinformqueue($vminformqueue)
        $this.testaddedalgorithm($vminformqueue)
        $this.resetinformqueue($vminformqueue)
        $this.testmismatchtaskid($vminformqueue)
        $this.testdoubletaskid($vminformqueue)
        $this.resetinformqueue($vminformqueue)
        $this.testgitstatus($vminformqueue)  
        #
        Write-Host '.'
        #
    }
    #
    [void]resetinformqueue($vminformqueue){
        $vminformqueue.removefile($vminformqueue.mainqueuelocation())
        $vminformqueue.removefile($this.basepath + '\upkeep_and_progress\inForm_queue.csv')
        $vminformqueue.copy($this.copypath, ($this.mpath + $vminformqueue.mainqueue_path))
    }
    #
    [void]testvminformqueuecontstructors(){
        #
        write-host '.'
        Write-Host 'test [vminformqueue] constructors started'
        #
        try{
            vminformqueue -mpath $this.mpath | Out-Null
        } catch {
            Throw ('[vminformqueue] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try{
            vminformqueue -mpath $this.mpath -project '0' | Out-Null
        } catch {
            Throw ('[vminformqueue] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #
        Write-Host 'test [vminformqueue] constructors finished'
        #         
    }
    #
    [PSCustomObject]testprojectimport($vminformqueue){
        #
        $projects = $vminformqueue.getapprojects('vminform')
        #
        $out = Compare-Object -ReferenceObject $projects -DifferenceObject @('0','1')
        #
        if ($out){
            Throw 'projects intialized not correctly'
        }
        #
        return $projects
        #
    }
    #
    [void]testopenmaininformqueue($vminformqueue){
        #
        write-host '.'
        write-host 'test open main inform queue started'
        $vminformqueue.openmainqueue()
        if (!$vminformqueue.maincsv){
            Throw 'main queue is empty'
        }
        write-host '    main queue:'
        write-host ($vminformqueue.maincsv | Format-Table | Out-String)
        #
        $currentprojecttasks = $vminformqueue.maincsv -match ('T' + $this.project.PadLeft(3,'0'))
        write-host '    current main tasks:'
        write-host ($currentprojecttasks | Format-Table | Out-String)
        #
        write-host 'open main queue finished'
    }
    #
    [void]testopenmaininformqueue($vminformqueue, $v2){
        #
        write-host '.'
        write-host 'test open main inform queue started'
        $vminformqueue.openmainqueue($false)
        if ($vminformqueue.maincsv){
            Throw 'main queue is not empty'
        }
        #
        write-host 'open main queue finished'
    }
    #
    [void]testopenlocalqueue($vminformqueue){
        #
        Write-host '.'
        Write-Host 'Tests for [vminformqueue] local queue started'
        #
        $vminformqueue.openmainqueue($false)
        #
        $localqueuefile = $vminformqueue.localqueuelocation($this.project)
        write-host '    local queue path:' $localqueuefile
        if (!(split-path $localqueuefile | test-path)){
            Throw ('local queue file path not found: ' + $localqueuefile)
        }
        #
        $vminformqueue.getlocalqueue($this.project, $false)
        #
        write-host ($vminformqueue.localqueue.($this.project) | Format-Table | Out-String)
        #
        if ($vminformqueue.localqueue.($this.project)){
            Throw 'local queue not read as empty intially'
        }
        #
        Write-host 'test for local queue finished'
    }
    #
    [void]testopenlocalqueue($vminformqueue, $v2){
        #
        Write-host '.'
        Write-Host 'Tests for [vminformqueue] local queue started'
        #
        $vminformqueue.openmainqueue($false)
        #
        $localqueuefile = $vminformqueue.localqueuelocation($this.project)
        write-host '    local queue path manual:' $localqueuefile
        if (!(split-path $localqueuefile | test-path)){
            Throw ('local queue file path not found: ' + $localqueuefile)
        }
        #
        $vminformqueue.getlocalqueue($this.project, $false)
        #
        write-host '    local queue path from vm obj:' $vminformqueue.localqueuefile.($this.project)
        if (!(split-path $vminformqueue.localqueuefile.($this.project) | test-path)){
            Throw ('local queue file path not found: ' + $localqueuefile)
        }
        #
        write-host 'local queue:'
        write-host ($vminformqueue.localqueue.($this.project) | Format-Table | Out-String)
        #
        if (!$vminformqueue.localqueue.($this.project)){
            Throw 'local queue read as empty intially'
        }
        #
        Write-host 'test for local queue finished'
    }
    #
    [void]testcoalescemethods($vminformqueue){
        #
        Write-host '.'
        Write-Host 'Tests for [vminformqueue] internal methods starting'
        #
        $vminformqueue.openmainqueue($false)
        $currentprojecttasks = $vminformqueue.maincsv -match ('T' + $this.project.PadLeft(3,'0'))
        $vminformqueue.openlocalqueue($this.project, $false)
        #
        write-host '    updating local queue'
        $vminformqueue.updatelocalvminformqueue($currentprojecttasks, $this.project)
        $this.checklocalqueue($vminformqueue, 5)
        #
        write-host '    update main queue'
        $vminformqueue.updatemainvminformqueue($this.project)
        write-host ($vminformqueue.maincsv | Format-Table | Out-String)
        if ($vminformqueue.maincsv.Length -ne 11){
            Throw 'main queue tasks were added with no new local tasks'
        }
        #
        $vminformqueue.writelocalqueue($this.project)
        $vminformqueue.writemainqueue($vminformqueue.mainqueuelocation())
        #
        if ((get-filehash $vminformqueue.mainqueuelocation()).hash `
                -ne (get-filehash $this.copypath).hash){
            Throw 'Changes made to the main queue before they were supposed to'
        }
        #
        Write-Host 'Tests for [vminformqueue] internal methods finished'
    }
    #
    [void]testcoalescemethods($vminformqueue, $v2){
        #
        Write-host '.'
        Write-Host 'Tests for [vminformqueue] internal methods starting'
        #
        $vminformqueue.openmainqueue($false)
        $currentprojecttasks = $vminformqueue.maincsv -match ('T' + $this.project.PadLeft(3,'0'))
        $vminformqueue.openlocalqueue($this.project, $false)
        #
        write-host '    updating local queue'
        $vminformqueue.updatelocalvminformqueue($currentprojecttasks, $this.project)
        $this.checklocalqueue($vminformqueue, 5)
        #
        write-host '    update main queue'
        $vminformqueue.updatemainvminformqueue($this.project)
        $this.checkmainqueue($vminformqueue, 5)
        #
        $vminformqueue.writelocalqueue($this.project)
        $vminformqueue.writemainqueue($vminformqueue.mainqueuelocation())
        #
        Write-Host 'Tests for [vminformqueue] internal methods finished'
    }
    #
    [void]testfullcoalesce($vminformqueue){
        #
        write-host '.'
        Write-Host 'Test full colaesce method started'
        #
        $vminformqueue.getlocalqueue($this.project, $false)
        $this.checklocalqueue($vminformqueue, 0)
        $vminformqueue.openmainqueue($false)
        $this.checkmainqueue($vminformqueue, 11)
        #
        try{
            $vminformqueue.coalescevminformqueues($this.project)
        } catch {
            Throw ('could not do the intial coalation of the inform queue. ' + $_.Exception.Message)
        }
        $this.checklocalqueue($vminformqueue, 5)
        #
        if ((get-filehash $vminformqueue.mainqueuelocation()).hash `
            -ne (get-filehash $this.copypath).hash){
            Throw 'Changes made to the main queue before they were supposed to'
        }
        Write-Host 'Test full colaesce method finished'
        #
    }
    #
    [void]checklocalqueue($vminformqueue, $n){
        #
        write-host ($vminformqueue.localqueue.($this.project) | Format-Table | Out-String)
        write-host '    table length:' ($vminformqueue.localqueue.($this.project)).length
        #
        if (($vminformqueue.localqueue.($this.project)).count -lt $n){
            Throw "local queue not updated for all $n lines"
        }
        #
        if (($vminformqueue.localqueue.($this.project)).count -gt $n){
            Throw 'local queue updated for too many lines for project 0'
        }
        #
    }
    #
    [void]checkmainqueue($vminformqueue, $n){
        write-host ($vminformqueue.maincsv | Format-Table | Out-String)
        if ($vminformqueue.maincsv.Length -ne $n){
            Throw 'main queue tasks are not correct'
        }
    }
    #
    # test if a task is added to the local queue without the algorthim
    # it does not get copied over
    #
    [void]testaddinformtask($vminformqueue){
        #
        Write-host '.'
        write-host 'check that new local tasks are written to local w/o alg started'
        #
        $vminformqueue.coalescevminformqueues($this.project)
        $vminformqueue.checkfornewtask($this.project, $this.slideid, 'NewAB')
        $this.checklocalqueue($vminformqueue, 6)
        #
        $vminformqueue.openmainqueue()
        $vminformqueue.updatemainvminformqueue($this.project)
        $this.checkmainqueue($vminformqueue, 11)
        #
        write-host 'check that new local tasks are written to local w/o alg finished'
    }
    #
    [void]testaddedalgorithm($vminformqueue){
        #
        write-host '.'
        write-host 'check that the new added algorithms get moved to the main queue started'
        #
        $vminformqueue.coalescevminformqueues($this.project)
        #
        $NewRow =  [PSCustomObject]@{
            TaskID = '99'
            SlideID = 'newslide'
            Antibody = 'Newab'
            Algorithm = 'newalg.ifp'
        } 
        $vminformqueue.localqueue.($this.project) += $NewRow
        $this.checklocalqueue($vminformqueue, 6)
        #
        $vminformqueue.pairqueues($this.project)
        $this.checkmainqueue($vminformqueue, 12)
        #
        write-host 'check that the new added algorithms get moved to the main queue finished'
        #
    }
    #
    [void]testmismatchtaskid($vminformqueue){
        #
        write-host '.'
        write-host 'test that the code handles local taskdid and main taskid mismatches started'
        #
        $vminformqueue.coalescevminformqueues($this.project)
        #
        $vminformqueue.localqueue.($this.project)[0].antibody = 'Newab'
        write-host ($vminformqueue.localqueue.($this.project) | Format-Table | Out-String)
        write-host ($vminformqueue.maincsv | Format-Table | Out-String)
        $vminformqueue.pairqueues($this.project)
        
        #
        $this.checklocalqueue($vminformqueue, 7)
        $this.checkmainqueue($vminformqueue, 13)
        #
        write-host 'test that the code handles local taskdid and main taskid mismatches finished'
        #
    }
    #
    [void]testdoubletaskid($vminformqueue){
        #
        write-host '.'
        write-host 'test that the code handles double local taskdid started'
        #
        $vminformqueue.coalescevminformqueues($this.project)
        #
        $NewRow =  [PSCustomObject]@{
            TaskID = '1'
            SlideID = 'newslide'
            Antibody = 'Newab'
            Algorithm = 'newalg.ifp'
        } 
        $vminformqueue.localqueue.($this.project) += $NewRow
        write-host ($vminformqueue.localqueue.($this.project) | Format-Table | Out-String)
        write-host ($vminformqueue.maincsv | Format-Table | Out-String)
        $vminformqueue.pairqueues($this.project)
        
        #
        $this.checklocalqueue($vminformqueue, 8)
        $this.checkmainqueue($vminformqueue, 14)
        #
        write-host 'test that the code handles double local taskdid finished'
        #
    }
    #
}
#
# launch test and exit if no error found
#
try{
    [testpsvminformqueue]::new() | Out-Null
} catch {
    Throw $_.Exception.Message
}
exit 0
