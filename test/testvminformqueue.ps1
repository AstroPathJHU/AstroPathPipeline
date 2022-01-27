 <# -------------------------------------------
 testvminformqueue
 created by: Benjamin Green - JHU
 Last Edit: 01.18.2022
 --------------------------------------------
 Description
 test the vm queue and how it performs
 -------------------------------------------#>
#
 Class testvminformqueue {
     #
    [string]$mpath 
    [string]$module 
    [string]$process_loc
    #
    testvminformqueue(){
        #
        $this.importmodule()
        $this.testvminformqueuecontstructors()
        $vminformqueue = vminformqueue -mpath $this.mpath
        $projects = $this.testprojectimport($vminformqueue)
        #
        $copypath = $vminformqueue.mpath + '\vminform-queue.csv'
        $vminformqueue.copy($vminformqueue.mainvminformqueuelocation(), $vminformqueue.mpath)
        #
        $this.testcoalescemethods($vminformqueue, $projects[0], $copypath)
        $this.testfullcoalesce($vminformqueue, $projects[0], $copypath)
        #
    }
    #
    importmodule(){
        $this.module = $PSScriptRoot + '/../astropath'
        Import-Module $this.module -EA SilentlyContinue
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing'
    }
    #
    [void]testvminformqueuecontstructors(){
        #
        Write-Host 'test [vminformqueue] constructors started'
        #
        try{
            $vminformqueue = vminformqueue -mpath $this.mpath
        } catch {
            Throw ('[vminformqueue] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try{
            $vminformqueue = vminformqueue -mpath $this.mpath -project '0'
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
    [void]testcoalescemethods($vminformqueue, $project, $copypath){
        #
        Write-Host 'Tests for [vminformqueue] internal methods starting'
        #
        $project = $project.ToString()
        #
        $mainqueue = $vminformqueue.openmainvminformqueue()
        $currentprojecttasks = $mainqueue -match ('T' + $project.PadLeft(3,'0'))
        #
        $localqueuefile = $vminformqueue.localvminfomqueuelocation($project)
        if (!(split-path $localqueuefile | test-path)){
            Throw ('local queue file path not found: ' + $localqueuefile)
        }
        $vminformqueue.localvminformdates.($project) = $vminformqueue.LastWrite($localqueuefile)
        $localqueue = $vminformqueue.openlocalvminformqueue($localqueuefile)
        #
        if ($localqueue){
            Throw 'local queue not read as empty intially'
        }
        #
        $localqueue = $vminformqueue.updatelocalvminfomqueue($currentprojecttasks, $localqueue)
        #
        if ($localqueue.Length -lt 6){
            Throw 'local queue not updated for all 6 lines'
        }
        #
        if ($localqueue.Length -gt 6){
            Throw 'local queue updated for too many lines for project 0'
        }
        #
        $mainqueue = $vminformqueue.updatemainvminfomqueue($project, $mainqueue, $localqueue)
        if ($mainqueue.Length -ne 6){
            Throw 'main queue tasks were added with no new local tasks'
        }
        #
        $vminformqueue.writelocalqueue($localqueue, $localqueuefile)
        $vminformqueue.writemainqueue($mainqueue, $vminformqueue.mainvminformqueuelocation())
        #
        if ((get-filehash $vminformqueue.mainvminformqueuelocation()).hash `
                -ne (get-filehash $copypath).hash){
            Throw 'Changes made to the main queue before they were supposed to'
        }
        #
        Write-Host 'Tests for [vminformqueue] internal methods finished'
    }
    #
    [void]testfullcoalesce($vminformqueue, $project, $copypath){
        #
        Write-Host 'Test full colaesce method started'
        #
        try{
            $vminformqueue.coalescevminformqueues($project)
        } catch {
            Throw ('could not do the intial coalation of the inform queue. ' + $_.Exception.Message)
        }
        #
        if ((get-filehash $vminformqueue.mainvminformqueuelocation()).hash `
        -ne (get-filehash $copypath).hash){
            Throw 'Changes made to the main queue before they were supposed to'
        }
        Write-Host 'Test full colaesce method finished'
        #
    }
    #
    [void]testfullcoalesce($vminformqueue, $project){
        #
        Write-Host 'Test full colaesce method 2 started'
        #
        try{
            $vminformqueue.coalescevminformqueues($project)
        } catch {
            Throw ('could not do the intial coalation of the inform queue. ' + $_.Exception.Message)
        }
        #
        $mainqueue = $vminformqueue.openmainvminformqueue()
        if ($mainqueue -ne 12){
            Throw 'vminform queue failed to write all 12 tasks back out'
        }
        Write-Host 'Test full colaesce method 2 finished'
        #
    }
    #
    [void]testaddedtasks($vminformqueue, $project, $copypath){
        Write-Host 'test if the queue updates correctly with tasks from other projects on it'
        #
        $newtask = @()
        $mainqueue = $vminformqueue.openmainvminformqueue()
        #
        $mainqueue | foreach-object {
            $newrow = $_ | 
                select TaskID, Specimen, Antibody, Algorithm, ProcessingLocation,StartDate, localtaskid
            $newrow.TaskID = 'T0010000' + $newrow.localtaskid 
            $newtask  += $newrow
        }
        #
        $mainqueue += $newtask
        #
        $currentprojecttasks = $mainqueue -match ('T' + $project.PadLeft(3,'0'))
        #
        $localqueuefile = $vminformqueue.localvminfomqueuelocation($project)
        $vminformqueue.localvminformdates.($project) = $vminformqueue.LastWrite($localqueuefile)
        $localqueue = $vminformqueue.openlocalvminformqueue($localqueuefile)
        #
        if ($localqueue.Length -ne 6){
            Throw 'local queue not read for all 6 lines'
        }
        #
        $localqueue = $vminformqueue.updatelocalvminfomqueue($currentprojecttasks, $localqueue)
        #
        if ($localqueue.Length -lt 6){
            Throw 'local queue not updated for all 6 lines'
        }
        #
        if ($localqueue.Length -gt 6){
            Throw 'local queue updated for too many lines for project 0'
        }
        #
        $mainqueue = $vminformqueue.updatemainvminfomqueue($project, $vminformqueue.mainvminformcsv, $localqueue)
        if ($mainqueue.Length -ne 12){
            Throw 'main queue tasks were added\ removed with no new local tasks'
        }
        #
        $vminformqueue.writelocalqueue($localqueue, $localqueuefile)
        $vminformqueue.writemainqueue($mainqueue, $vminformqueue.mainvminformqueuelocation())
        #
        $mainqueue2 = get-content $vminformqueue.mainvminformqueuelocation()
        #
        if ($mainqueue2.Length -ne 12){
            Throw 'vminform queue failed to write all 12 tasks back out correctly'
        }
        #
        if ($mainqueue2 -ne $mainqueue){
            Throw 'vminform queue failed to write all 12 tasks back out correctly'
        }
        #
        Write-Host 'test new tasks finished'
        #
    }
    #
    # test that the inform queue updates two folders correctly
    # 
    [void]testfullcoalesce($vminformqueue){
        #
        Write-Host 'Test full colaesce method 2 started'
        #
        try{
            $vminformqueue.coalescevminformqueues()
        } catch {
            Throw ('could not do the intial coalation of the inform queue. ' + $_.Exception.Message)
        }
        #
        $mainqueue = $vminformqueue.openmainvminformqueue()
        if ($mainqueue -ne 12){
            Throw 'vminform queue failed to write all 12 tasks back out'
        }
        #
        if ($vminformqueue.mainvminformcsv -ne 12){
            Throw 'vminform queue failed to collect all the tasks'
        }
        if ($vminformqueue.mainvminformcsv -ne $mainqueue){
            Throw 'vminform queues do not match?'
        }
        #
        Write-Host 'Test full colaesce method 2 finished'
        #
    }
    #
    # test if a task is added to the local queue without the algorthim
    # it does not get copied over
    #

    #
    # test if a task is on both queues with the same number but
    # tasks that don't match that the 
    # number on the local queue gets modified
    #

    #
    [void]cleanup($vminformqueue, $copypath, $projects){
        $vminformqueue.copy($copypath, $vminformqueue.mainvminformqueuelocation())
        $projects | ForEach-Object {
            $vminformqueue.removefile($vminformqueue.localvminfomqueuelocation($projects))
        }
    }

 }