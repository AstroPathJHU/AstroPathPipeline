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
        $r = $project_dat.dpath -replace( '/', '\')
        if ($r[0] -ne '\'){
            $root = '\\' + $project_dat.dpath 
        } else{
            $root = $project_dat.dpath
        }
        #
        $this.basepath = $root, '\', $project_dat.dname -join ''
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
        $vers = $this.GetVersion($this.mpath, $cmodule, $this.project)
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
    [string]im3mainfolder(){
        $path = $this.basepath + '\' + $this.slideid + '\im3'
        return $path
    }
    #
    [string]upkeepfolder(){
        $path = $this.basepath + '\upkeep_and_progress'
        return $path
    }
    #
    [string]Scan(){
        $path = $this.basepath + '\' + $this.slideid + '\im3\Scan*'
        $paths = get-childitem $path
        $scan = $paths | 
            select-object *, @{n = "IntVal"; e = {[int]$_.Name.substring(4)}} |
            sort-object IntVal |
            Select-Object -Last 1
        return $scan.Name
    }
    #
    [string]Scanfolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\im3\'+$this.Scan()
        return $path

    }
    #
    [string]qptifffile(){
        $path = $this.Scanfolder() + '\' + $this.slideid + 
            '_' + $this.Scan() + '.qptiff'
        return $path
    }
    #
    [string]annotationxml(){
        $path = $this.Scanfolder() + '\' + $this.slideid + '_' + 
            $this.Scan() + '_annotations.xml'
        return $path
    }
    #
    [string]batchIDfile(){
        $path = $this.Scanfolder() + '\BatchID.txt'
        return $path
    }
    #
    [string]flatfieldfolder(){
        $path = $this.basepath +'\flatfield'
        return $path
    }
    #
    [string]batchflatfield(){
        $path = $this.basepath +'\flatfield\flatfield_BatchID_' + 
            $this.BatchID + '.bin'
        return $path
    }
    #
    [string]pybatchflatfield(){
        $ids = $this.ImportCorrectionModels($this.mpath)
        if ($this.slideid -notcontains $this.batchid){
            $file = ($ids | Where-Object { $_.slideid `
                    -contains $this.slideid}).FlatfieldVersion
        } else  {
            $file1 = ($ids |
                Where-Object { $_.BatchID.padleft(2, '0') -contains $this.batchid `
                -and $_.Project -contains $this.project }).FlatfieldVersion
           if ($file1.Count -ne 1){
                $file = $file1[0]
           } elseif ($file1.Count -eq 1){
               $file = $file1
           } else {
               $file = ''
           }
        }
        return $file
    }
    #
    [string]pybatchflatfieldfullpath(){
          $flatfield = $this.mpath + '\flatfield\flatfield_' +
           $this.pybatchflatfield() + '.bin'
          return $flatfield
    }
    #
    [string]CheckSumsfile(){
        $path = $this.Scanfolder() + '\CheckSums.txt'
        return $path
    }
    #
    [string]im3folder(){
        $path = $this.Scanfolder() + '\MSI'
        return $path 
    }
    #
    [string]informfolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\inform_data'
        return $path

    }
    #
    [string]componentfolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\inform_data\Component_Tiffs'
        return $path
    }
    #
    [string]segmapfolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\inform_data\Component_Tiffs'
        return $path
    }
    #
    [string]phenotypefolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\inform_data\Phenotyped'
        return $path

    }
    #
    [string]cantibodyfolder(){
        $path = $this.phenotypefolder() + '\'  + $this.cantibody
        return $path
    }
    #
    [string]mergefolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\inform_data\Phenotyped\Results\Tables'
        return $path

    }
    #
    [string]xmlfolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\im3\xml'
        return $path

    }
    #
    [string]exposurexmlfolder(){
        return $this.xmlfolder()
    }
    #
    [string]meanimagefile(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\im3\' + $this.slideid + '-mean.flt'
        return $path
    }
    #
    [string]meanimagefolder(){
        $path = $this.im3mainfolder() + '\meanimage'
        return $path
    }
    #
    [string]flatwim3folder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\im3\flatw'
        return $path
    }
    #
    [string]flatwfolder(){
        $path = '\\'+$this.project_data.fwpath + '\' + 
            $this.slideid
        return $path
    }
    #
    [string]fwfolder(){
        return $this.flatwfolder()
    }
    #
    [string]fw01folder(){
        return $this.flatwfolder()
    }
    #
    [string]mergeconfigfile(){
        $path = $this.basepath + '\Batch\MergeConfig_' + 
            $this.BatchID
        return $path
    }
    #
    [string]warpoctetsfolder(){
        $file2 = $this.basepath, '\', $this.slideid,
            '\im3\warping\octets' -join ''
        return $file2
    }
    #
    [string]warpoctetsfile(){
        $file2 = $this.warpoctetsfolder(),
            '\', $this.slideid, '-all_overlap_octets.csv' -join ''
        return $file2
    }
    #
    [string]warpbatchfolder(){
        $path = $this.basepath +'\warping\Batch_' + $this.BatchID
        return $path
    }
    #
    [string]warpprojectfolder(){
        $path = $this.mpath +'\warping\Project_' + $this.project
        return $path
    }
    #
    [string]warpbatchoctetsfolder(){
        $path = $this.basepath +'\warping\Batch_' +
            $this.BatchID + '\octets'
        return $path
    }
    #
    [string]warpprojectoctetsfolder(){
        $path = $this.basepath +
            '\warping\Project_' + $this.project + '\octets'
        return $path
    }
    #
    [void]findantibodies(){
        $this.findantibodies($this.basepath)
    }
    #
    [void]findantibodies($basepath){
        #
        $this.ImportMergeConfig($basepath)
        $targets = $this.mergeconfig_data.Target
        $qa = $this.mergeconfig_data.ImageQA.indexOf('Tumor')
        #
        if ($qa -ge 0){
            $targets[$qa] = 'Tumor'
        }
        #
        $this.antibodies = $targets
        #
    }
    #
    [int]getcount($source, $forceupdate){
        #
        if ($forceupdate){
            $cnt = ($this.getfiles(
                $source, $forceupdate)).Count
        } else {
            $cnt = ($this.getfiles(
                $source)).Count
        }
        #
        return $cnt
        #
    }
    #
    [int]getmindate($source, $forceupdate){
        #
        if ($forceupdate){
            $dates = ($this.getfiles(
                $source, $forceupdate)).LastWriteTime
        } else {
            $dates = ($this.getfiles(
                $source)).LastWriteTime
        }
        #
        $date = ($dates | Measure-Object -Minimum).Minimum
        #
        return $date
        #
    }
    #
    [int]getmaxdate($source, $forceupdate){
        #
        if ($forceupdate){
            $dates = ($this.getfiles(
                $source, $forceupdate)).LastWriteTime
        } else {
            $dates = ($this.getfiles(
                $source)).LastWriteTime
        }
        #
        $date = ($dates | Measure-Object -Maximum).Maximum
        #
        return $date
        #
    }
    #
    [system.object]getfiles($source){
        #
        if (!$this.($source + 'files')){
            $this.getfiles($source, $false) | Out-Null
        }
        #
        return $this.($source + 'files')
        #
    }
    #
    [system.object]getfiles($source, $forceupdate){
        #
        $this.($source + 'files') = $this.listfiles(
            $this.($source + 'folder')(), $this.($source + 'constant')
        )
        #
        return $this.($source + 'files')
        #
    }
    #
    [array]getnames($source, $type, $forceupdate){
        #
        if ($forceupdate){
            $names = ($this.getfiles(
                $source, $forceupdate)).($type)
        } else {
            $names = ($this.getfiles(
                $source)).($type)
        }
        #
        return $names
        #
    }
    #
}
