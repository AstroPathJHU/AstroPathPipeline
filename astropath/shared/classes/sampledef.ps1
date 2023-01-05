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
    [system.object]$im3files
    [system.object]$xmlfiles
    [system.object]$exposurexmlfiles
    [system.object]$fwfiles
    [system.object]$fw01files
    [system.object]$flatwim3files
    [system.object]$segmapfiles
    [system.object]$mergefiles
    [system.object]$cantibodyfiles
    #
    [string]$im3constant = '.im3'
    [string]$fwconstant = '.fw'
    [string]$fw01constant = '.fw01'
    [string]$rawconstant = '.Data.dat'
    [string]$flatwim3constant = '.im3'
    [string]$xmlconstant = '.xml'
    [string]$exposurexmlconstant = '.SpectralBasisInfo.Exposure.xml'
    [string]$algorithmconstant = '.ifp'
    [string]$projectconstant = '.ifp'
    [string]$segmapconstant = '_component_data_w_seg.tif'
    [string]$mergeconstant = '_cleaned_phenotype_data.csv'
    [string]$cellsegconstant = '_cell_seg_data.txt'
    [string]$binsegconstant = '_binary_seg_maps.tif'
    [string]$cellsegsumconstant = '_cell_seg_data_summary.tif'
    [string]$componentconstant = '_component_data.tif'
    [string]$cantibodyconstant = '_cell_seg_data.txt'
    #
    [array]$antibodies
    [array]$binarysegtargets
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
        $this.importslide($this.mpath)
        $this.Sample($slideid, $this.slide_data)
    }
    #
    sampledefbatch($batchid, $project){
        $this.project = $project
        $this.importslide($this.mpath)
        $this.Batch($batchid, $this.slide_data)
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
        if ($slideid -match 'Control_TMA') {
            return
        }
        $slide = $slides | & { process { 
            if ($_.SlideID -eq $slideid.trim()){ $_ }
        }}
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
        $batch = $slides | & {process { 
            if (
                $_.BatchID -eq $mbatchid.trim() -and 
                $_.Project -eq $this.project.trim()
            ) { $_ }
        }}
        #
        if (!$batch){
            Throw ('Not a valid batchid: project - ' + $this.project +
                 '; batchid - ' + $mbatchid.trim() + '; module - ' + $this.module)
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
        $this.importcohortsinfo($this.mpath)
        #
        $this.project_data = $this.full_project_dat | & { process {
            if ($_.Project -eq $this.project) {$_}
        }}
        #
        $this.basepath = (
            $this.uncpaths($this.project_data.dpath),
            $this.project_data.dname -join '\'
        )
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
        $this.moduleinfo.($cmodule) = @{mainlog=$cmainlog; slidelog=$cslidelog; version=$vers}
        #
        $cmainlog = $null
        $cslidelog = $null
        $vers = $null
        #
    }
    #
    [string]slidelogfolder(){
        return ($this.basepath + '\' + $this.slideid + '\logfiles')
    }
    #
    [string]slidelogbase(){
        return ($this.slidelogfolder() +
            '\' + $this.slideid + '-' )
    }
    #
    [string]slidelogbase($cmodule){
        return ($this.slidelogfolder() + 
            '\' + $this.slideid + '-' + $cmodule + '.log')
    }
    #
    [string]mainlogfolder(){
        return ($this.basepath + '\logfiles' )

    }
    #
    [string]mainlogbase(){
        return ($this.mainlogfolder() + '\' )
    }
    #
    [string]mainlogbase($cmodule){
        return ($this.mainlogfolder() + '\' + $cmodule + '.log')
    }
    #
    [void]findsegmentationtargets(){
        #
        $this.ImportMergeConfigCSV($this.basepath, $this.batchid)
        #
        $sorted = $this.mergeconfig_data | 
            Where-Object {$_.SegmentationStatus -gt 0} | 
            Sort-Object -Property Opal
        #
        $this.binarysegtargets = $this.mergeconfig_data | 
            Where-Object {
                $_.SegmentationStatus -gt 0 -and
                $_.TargetType -eq 'Lineage'
            } | 
            Group-Object SegmentationStatus | 
            foreach-object {
                $lowestopal = ($_.group | Sort-Object -Property Opal)[0]
                if ($lowestopal.ImageQA -eq 'Tumor') {
                    $lowestopal.Target = 'Tumor'
                }
                $lowestopal
            }
        #
    }
    #
}
