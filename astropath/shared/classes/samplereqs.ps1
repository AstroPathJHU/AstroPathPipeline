<# -------------------------------------------
 samplereqs
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
    methods used to describe metadata of what
    files are required for modules
 -------------------------------------------#>
 class samplereqs : sampledef {
    # 
    [array]$transferreqfiles = @('CheckSums.txt', '_annotations.xml', '.qptiff', 'BatchID.txt')
    [array]$xmlreqfiles = @('Full.xml','Parameters.xml', '.exposurexml')
    [array]$meanimagereqfilesv1 = @('-mean.csv', '-mean.flt')
    [array]$meanimagereqfiles = @('-sum_images_squared.bin',
        '-std_err_of_mean_image.bin',
        '-mean_image.bin')
    [array]$warpoctetsreqfiles = @('-all_overlap_octets.csv')
    [array]$batchwarpkeysreqfiles = @('final_pattern_octets_selected.csv', 'initial_pattern_octets_selected.csv',
        'image_keys_needed.txt','principal_point_octets_selected.csv')
    [array]$batchwarpfitsreqfiles = @('weighted_average_warp.csv')
    [array]$imagecorrectionreqfiles = @('.fw', '.fw01','.flatwim3')
    [array]$segmentationreqfiles = @('.segmap')
    #
    samplereqs(){}
    samplereqs($mpath) : base($mpath){}
    samplereqs($mpath, $module) : base($mpath, $module){}
    samplereqs($mpath, $module, $slideid) : base($mpath, $module, $slideid){}
    samplereqs($mpath, $module, $batchid, $project) : base($mpath, $module, $batchid, $project){}
    #
    [switch]testtransferfiles(){
        return $this.testfiles($this.scanfolder(), $this.transferreqfiles)
    }
    #
    [switch]testxmlfiles(){
        #
        return $this.testfiles($this.xmlfolder(), 
            $this.im3constant, $this.xmlreqfiles)
        <#
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
        #>
    }
    #
    [void]testim3folder(){
        if (!(test-path $this.im3folder())){
            Throw "im3 folder not found for:" + $this.im3folder()
        }
    }
    #
    [switch]testmeanimagefiles(){
        #
        if ($this.vers -match '0.0.1'){
            #
            return $this.testfiles($this.im3folder(), $this.meanimagereqfilesv1)
            #
        } else {
            #
            return $this.testfiles($this.meanimagefolder(), $this.meanimagereqfiles)
            #
        }
        #
    }
    #
    [switch]testbatchmicompfiles(){
        #
        if ($this.teststatus){
            $micomp_data = $this.ImportMICOMP($this.mpath, $false)
        } else {
            $micomp_data = $this.ImportMICOMP($this.mpath)
        }
        #
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
    [switch]testimagecorrectionfiles(){
        #
        return $this.testfiles($this.flatwfolder(), 
            $this.im3constant, $this.imagecorrectionreqfiles())
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
        #
        $p = ($this.meanimagefolder() + '\' + $this.slideid + '-mask_stack.bin')
        #
        if (!(test-path $p)){
            return $true
        }
        #
        return $this.testwarpoctetsfiles()
        #
    }
    #
    [switch]testwarpoctetsfiles(){
        #
        return $this.testfiles($this.warpoctetsfolder(), $this.warpoctetsreqfiles)
        #
    }
    #
    [switch]testbatchwarpkeysfiles(){
        #
        return $this.testfiles($this.warpbatchoctetsfolder(), $this.batchwarpkeysreqfiles)
        #
    }
    #
    [switch]testbatchwarpfitsfiles(){
        #
        return $this.testfiles($this.warpbatchfolder(), $this.batchwarpfitsreqfiles)
        #
    }
    #
    [switch]testsegmentationfiles(){
        #
        return $this.testfiles($this.componentfolder(),
             $this.im3constant, $this.segmentationreqfiles)
        #
    }
    #
    [switch]testfiles($path, $testfiles){
        #
        if (!(test-path $path)){
            return $false
        }
        #
        foreach ($file in $testfiles) {
            $s = $this.testfile($path, $file)
            if (!$s){
                return $false
            }
        }
        #
        return $true
        #
    }
    #
    [switch]testfiles($path, $source, $testfiles){
        #
        if (!(test-path $path)){
            return $false
        }
        #
        foreach ($file in $testfiles) {
            if ($file[0] -match '\.'){
                $s = $this.testfilecount($source, $file)
            } else {
                $s = $this.testfile($path, $file)
            }
            if (!$s){
                return $false
            }
        }
        #
        return $true
        #
    }
    #
    [switch]testfile($path, $file){
        #
        if ($file[0] -match '-'){
            $file = $this.slideid + $file
        }
        #
        if (($file[0] -match '_') -or
            ($file -match '.qptiff')){
            $file = $this.slideid + '_' + $this.Scan() +  $file
        }
        #
        if (($file -match 'Parameters') -or
            ($file -match 'Full')){
            $file = $this.slideid + '.' + $file
        }
        #
        $fullpath = $path + '\' + $file
        if (!(test-path $fullpath)){
            return $false
        }
        #
        return $true
        #
    }
    #
    [switch]testfilecount($source, $testfile){
        #
        $source = $source -replace '\.', ''
        $testfile = $testfile -replace '\.', ''
        #
        if ($this.getcount($source, $true) -ne `
            $this.getcount($testfile, $true) 
        ){
            return $false
        }
        #
        return $true
        #
    }
 }
 #