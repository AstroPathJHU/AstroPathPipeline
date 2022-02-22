<# -------------------------------------------
 sampledef
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to build a sample object
 -------------------------------------------#>
class sampledef : sharedtools{
    [string]$cohort
    [string]$project
    [string]$BatchID
    [string]$basepath
    [PSCustomObject]$project_data
    [PSCustomObject]$batchslides
    [string]$mainlog
    [string]$slidelog
    [hashtable]$moduleinfo = @{}
    #
    sampledef(){}
    #
    sampledef($mpath, $module){
        $this.mpath = $mpath
        $this.module = $module 
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
        $this.Sample($slideid, $this.mpath, $slides)
    }
    #
    sampledefbatch($batchid, $project){
        $this.project = $project
        $slides = $this.importslideids($this.mpath)
        $this.Batch($batchid, $this.mpath, $slides)
    }
    #
    Sample(
        [string]$slideid="",
        [string]$mpath,
        [PSCustomObject]$slides
    ){
        $this.ParseAPIDdef($slideid, $slides)
        $this.defbase($mpath)
    }
    #
    Batch(
        [string]$batchid="",
        [string]$mpath,
        [PSCustomObject]$slides
    ){
        $this.ParseAPIDdefbatch($batchid, $slides)
        $this.defbase($mpath)
    }
    #
    [void]ParseAPIDdef([string]$slideid, [PSCustomObject]$slides){
        $slide = $slides | 
                Where-Object -FilterScript {$_.SlideID -eq $slideid.trim()}
        #
        if (!$slide){
            Throw ($slideid.trim() + ' is not a valid slideid. Check the APID tables and\or confirm the SlideID.')
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
            $mbatchid = $mbatchid[1]
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
    [void]defbase([string]$mpath){
        $this.mpath = $mpath
        $project_dat = $this.importcohortsinfo($this.mpath)
        $project_dat = $project_dat | 
                Where-Object -FilterScript {$_.Project -eq $this.project}
        <#
                #Adjust if testing on jenkins
        if ($project_dat.dpath -match '/var/lib/jenkins') {
            $this.basepath = $project_dat.dpath + '/' + $project_dat.dname
        }
        else {
            $this.basepath = '\\' + $project_dat.dpath + '\' + $project_dat.dname
        }
        more robust path editing #>
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
        $this.deflogpaths()
        #
    }
    #
    # define log paths
    #
    [void]deflogpaths(){
        #
        $this.mainlog = $this.basepath + '\logfiles\' + $this.module + '.log'
        $this.slidelog = $this.basepath + '\' + $this.slideid + '\logfiles\' +
             $this.slideid + '-' + $this.module + '.log'
        #
    }
    #
    [void]deflogpaths($cmodule){
        #
        $cmainlog = $this.basepath + '\logfiles\' + $cmodule + '.log'
        if ($cmodule -match 'batch'){
                $cslidelog = $cmainlog
            } else {
                $cslidelog = $this.basepath + '\' + $this.slideid + '\logfiles\' +
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
        $path = $this.slidelogfolder() + '\' + $this.slideid + '-'
        return $path
    }
    #
    [string]slidelogbase($cmodule){
        $path = $this.slidelogfolder() + '\' + $this.slideid + '-' + $cmodule + '.log'
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
    [string]im3folder(){
        $path = $this.basepath + '\' + $this.slideid + '\im3'
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
            $file = ($ids | Where-Object { $_.slideid -contains $this.slideid}).FlatfieldVersion
        } else  {
            $file1 = ($ids | Where-Object { $_.BatchID.padleft(2, '0') `
                -contains $this.batchid}).FlatfieldVersion
           if ($file1){
                $file = $file1[0]
           } else {
               $file = ''
           }
        }
        return $file
    }
    #
    [string]pybatchflatfieldfullpath(){
          $flatfield = $this.mpath + '\flatfield\flatfield_' + $this.pybatchflatfield() + '.bin'
          return $flatfield
    }
    #
    [string]CheckSumsfile(){
        $path = $this.Scanfolder() + '\CheckSums.txt'
        return $path
    }
    #
    [string]MSIfolder(){
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
    [string]phenotypefolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\inform_data\Phenotyped'
        return $path

    }
    #
    [string]xmlfolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\im3\xml'
        return $path

    }
    #
    [string]meanimagefile(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\im3\' + $this.slideid + '-mean.flt'
        return $path
    }
    #
    [string]meanimagefolder(){
        $path = $this.im3folder() + '\meanimage'
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
    [string]mergeconfigfile(){
        $path = $this.basepath + '\Batch\MergeConfig_' + 
            $this.BatchID
        return $path
    }
    #
    [void]testim3folder(){
        if (!(test-path $this.im3folder())){
            Throw "im3 folder not found for:" + $this.im3folder()
        }
    }
    #
    [switch]testbatchflatfield(){
        #
        if (!(test-path $this.batchflatfield())){
            return $false
        }
        #
        return $true
    }
    #
    [switch]testpybatchflatfield(){
        #
        if (!(test-path $this.pybatchflatfieldfullpath())){
            return $false
        }
        #
        return $true
    }
    #
    [switch]testxmlfiles(){
        #
        $xml = $this.xmlfolder()
        $im3s = get-childitem ($this.Scanfolder() + '\MSI\*') *im3
        $im3n = ($im3s).Count + 2
        #
        if (!(test-path $xml)){
            return $false
        }
        #
        # check xml files = im3s
        #
        $xmls = get-childitem ($xml + '\*') '*xml'
        $files = ($xmls).Count
        if (!($im3n -eq $files)){
            return $false
        }
        #
        return $true
        #
    }
    #
    [switch]testmeanimagefiles(){
        #
        if ($this.vers -match '0.0.1'){
            #
            # check for mean images
            # 
            $file = $this.im3folder() + '\' + $this.slideid + '-mean.csv'
            $file2 = $this.im3folder() + '\' + $this.slideid + '-mean.flt'
            #
            if (!(test-path $file)){
                return $false
            }
            if (!(test-path $file2)){
                return $false
            }
        } else {
            #
            # check for meanimage directory
            #
            $p = $this.meanimagefolder()
            if (!(test-path $p)){
                return $false
            }
            #
            $files = @('-sum_images_squared.bin', '-std_err_of_mean_image.bin', `
                     '-mean_image.bin') #'-mask_stack.bin',
            #
            foreach ($file in $files) {
                $fullpath = $p + '\' + $this.slideid + $file
                if (!(test-path $fullpath)){
                    return $false
                }
            }
            #
        }
        #
        return $true
        #
    }
    #
    [switch]testbatchmicompfiles(){
        $micomp_data = $this.ImportMICOMP($this.mpath)
        if (($micomp_data.root_dir_1 -contains ($this.basepath + '\') -AND
                $micomp_data.slide_ID_1 -contains ($this.slideid)) -OR
            (($micomp_data.root_dir_2 -contains ($this.basepath + '\') -AND
                $micomp_data.slide_ID_2 -contains ($this.slideid)))){
            return $true
        }
        #
        return $false
        #
    }
    #
    [switch]testimagecorrectionfiles(){
        #
        $im3s = (get-childitem ($this.Scanfolder() + '\MSI\*') *im3).Count
        #
        $paths = @($this.flatwim3folder(), $this.flatwfolder(), $this.flatwfolder())
        $filetypes = @('*im3', '*fw', '*fw01')
        #
        for ($i=0; $i -lt 3; $i++){
            #
            if (!(test-path $paths[$i])){
                return $false
            }
            #
            # check files = im3s
            #
            $files = (get-childitem ($paths[$i] + '\*') $filetypes[$i]).Count
            if (!($im3s -eq $files)){
                return $false
            }
        }
        #
        return $true
        #
    }
    #
    [switch]testsegmentationfiles(){
        #
        $table = $this.phenotypefolder() + '\Results\Tables'
        if (!(test-path $table + '\*csv')){
            return $false
        }
        $comp = (get-childitem ($table + '\*') '*csv').Count
        $seg = (get-childitem ($this.componentfolder() + '\*') '*data_w_seg.tif').Count
        if (!($comp -eq $seg)){
            return $false
        }
        return $true
        #
    }
    #
    [switch]testwarpoctets(){
        #
        $logfile = $this.basepath, '\', $this.slideid,
            '\logfiles\', $this.slideid, '-warpoctets.log' -join ''
        #
        if (test-path $logfile){
            $log = $this.importlogfile($logfile)
            if ($log.Message -match "Sample is not good"){
                return $true
            }
        }
        $p = ($this.meanimagefolder() + '\' + $this.slideid + '-mask_stack.bin')
        #
        if (!(test-path $p)){
            return $true
        }
        #
        $this.testwarpoctetsfiles($this.slideid)
        #
        return $true
    }
    #
    [switch]testwarpoctetsfiles(){
        #
        $file = $this.basepath, '\warping\octets\',
            $this.slideid, '-all_overlap_octets.csv' -join ''
       #
       $file2 = $this.basepath, '\', $this.slideid,
            '\im3\warping\octets\', $this.slideid, '-all_overlap_octets.csv' -join ''
       <#
       $file3 = $this.basepath, '\', $slideid,
        '\im3\warping\octets\image_keys_needed.txt' -join ''
       #>
        if (!(test-path $file) -AND !(test-path $file2)){
            return $false
        }
        <#
        if (!(test-path $file3)){
            return $false
        }
        #>
        return $true
    }
    #
}
