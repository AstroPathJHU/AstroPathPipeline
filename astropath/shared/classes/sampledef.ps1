<# -------------------------------------------
 sampledef
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build a sample object
 -------------------------------------------#>
class sampledef : sharedtools{
    #
    [string]$cohort
    [string]$project
    [string]$BatchID
    [string]$basepath
    [PSCustomObject]$project_data
    [PSCustomObject]$batchslides
    [string]$mainlog
    [string]$slidelog
    [string]$cantibody
    [hashtable]$moduleinfo = @{}
    #
    sampledef(){}
    #
    sampledef($mpath){
        $this.mpath = $mpath
        $this.defbase()
    }
    #
    sampledef($mpath, $module){
        $this.mpath = $mpath
        $this.module = $module 
        $this.defbase()
    }
    #
    sampledef($mpath, $module, $slideid){
        $this.mpath = $mpath
        $this.module = $module 
        $this.sampledefslide($slideid)
    }
    #
    sampledef($mpath, $module, $batchid, $project){
        $this.mpath = $mpath
        $this.module = $module 
        $this.sampledefbatch($batchid, $project)
    }
    #
    sampledefslide($slideid){
        $slides = $this.importslideids($this.mpath)
        $this.Sample($slideid, $slides)
    }
    #
    sampledefbatch($batchid, $project){
        $this.project = $project
        $slides = $this.importslideids($this.mpath)
        $this.Batch($batchid, $slides)
    }
    #
    Sample(
        [string]$slideid="",
        [PSCustomObject]$slides
    ){
        $this.ParseAPIDdef($slideid, $slides)
        $this.defbase()
        $this.deflogpaths()
    }
    #
    Sample(
        [string]$slideid="",
        [string]$module,
        [PSCustomObject]$slides
    ){
        $this.module = $module
        $this.ParseAPIDdef($slideid, $slides)
        $this.defbase() 
        $this.deflogpaths()
    }
    #
    Batch(
        [string]$batchid="",
        [PSCustomObject]$slides
    ){
        $this.ParseAPIDdefbatch($batchid, $slides)
        $this.defbase()
        $this.deflogpaths()
    }
    #
    [void]ParseAPIDdef([string]$slideid){
        $this.ParseAPIDdef($slideid, $this.slide_data)
        #
    }
    #
    [void]ParseAPIDdef([string]$slideid, [PSCustomObject]$slides){
        $slide = $slides | 
                Where-Object -FilterScript {$_.SlideID -eq $slideid.trim()}
        #
        if (!$slide){
            Throw ($slideid.trim() +
             ' is not a valid slideid. Check the APID tables and\or confirm the SlideID.')
        }
        $this.slideid = $slide.SlideID.trim()
        $this.project = $slide.Project
        $this.cohort = $slide.Cohort
        $this.BatchID = $slide.BatchID.padleft(2, '0')
        #
    }
    #
    [void]ParseAPIDdefbatch([string]$mbatchid){
        $this.ParseAPIDdefbatch($mbatchid, $this.slide_data)
        #
    }
    #
    [void]ParseAPIDdefbatch([string]$mbatchid, [PSCustomObject]$slides){
        #
        if ($mbatchid[0] -match '0'){
            [string]$mbatchid = $mbatchid[1]
        }
        #
        $batch = $slides | 
                Where-Object -FilterScript {$_.BatchID -eq $mbatchid.trim() -and 
                    $_.Project -eq $this.project.trim()}
        #
        if (!$batch){
            Throw 'Not a valid batchid'
        } elseif ($batch.Count -eq 1){
            $this.project = $batch.Project
            $this.cohort = $batch.Cohort
            $this.BatchID = $batch.BatchID.padleft(2, '0')
        } else{
            $this.project = $batch.Project[0]
            $this.cohort = $batch.Cohort[0]
            $this.BatchID = $batch.BatchID[0].padleft(2, '0')
        }
        #
        $this.slideid = $this.BatchID
        $this.batchslides = $batch
        #
    }
    #
    [void]defbase(){
        #
        $this.importcohortsinfo($this.mpath) | Out-Null
        #
        $project_dat = $this.full_project_dat| 
                    Where-Object -FilterScript {$_.Project -eq $this.project}
        #
        $root = $this.uncpaths($project_dat.dpath)
        #
        $this.basepath = $root, $project_dat.dname -join '\'
        #
        $this.project_data = $project_dat
        #
    }
    #
    [void]defbase([string]$mpath){
        $this.mpath = $mpath
        $this.defbase()
        #
    }
    #
    # define log paths
    #
    [void]deflogpaths(){
        #
        $this.mainlog = $this.basepath + '\logfiles\' +
            $this.module + '.log'
        if ($this.module -match 'batch'){
            $this.slidelog = $this.mainlog
        } else {
            $this.slidelog = $this.basepath + '\' +
                $this.slideid + '\logfiles\' +
                $this.slideid + '-' + $this.module + '.log'
        }
        #
    }
    #
    [void]deflogpaths($cmodule){
        #
        $cmainlog = $this.basepath + '\logfiles\' + $cmodule + '.log'
        if ($cmodule -match 'batch'){
                $cslidelog = $cmainlog
            } else {
                $cslidelog = $this.basepath + '\' +
                    $this.slideid + '\logfiles\' +
                    $this.slideid + '-' + $cmodule + '.log'
        }
        $vers = $this.GetVersion($this.mpath, $cmodule, $this.project, $true)
        $this.moduleinfo.($cmodule) = @{mainlog =$cmainlog; slidelog=$cslidelog; version=$vers}
        #
    }
    #
    [string]slidelogfolder(){
        $path = $this.basepath + '\' + $this.slideid + '\logfiles'
        return $path

    }
    #
    [string]slidelogbase(){
        $path = $this.slidelogfolder() +
            '\' + $this.slideid + '-'
        return $path
    }
    #
    [string]slidelogbase($cmodule){
        $path = $this.slidelogfolder() + 
            '\' + $this.slideid + '-' + $cmodule + '.log'
        return $path
    }
    #
    [string]mainlogfolder(){
        $path = $this.basepath + '\logfiles'
        return $path

    }
    #
    [string]mainlogbase(){
        $path = $this.mainlogfolder() + '\' 
        return $path
    }
    #
    [string]mainlogbase($cmodule){
        $path = $this.mainlogfolder() + '\' + $cmodule + '.log'
        return $path
    }
   #
}
