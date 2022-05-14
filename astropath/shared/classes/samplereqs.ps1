<# -------------------------------------------
 samplereqs
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
    methods used to describe metadata of what
    files are required for modules. when adding
    a module the pipeline add:
    - a [module]reqfiles array to this method. 
        - depending on the first letter of the 
        requirement decides how the code will 
        handle files:
        - files listed with '-' as the first 
        character will be appended with the 
        slideid in the search. 
        - a '.' indicates to handle as a list
        of files (for example '.im3' for im3 files)
        - no 
 -------------------------------------------#>
 class samplereqs : samplefiles {
    # 
    [array]$transferreqfiles = @('_annotations.xml', '.qptiff', 'BatchID.txt')
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
    [array]$vminformreqfiles = @('.cellseg', '.cellsegsum','.component')
    [array]$mergereqfiles = @('.merge')
    [array]$segmapsreqfiles = @('.segmap')
    #
    # regex's for prepending variables at the time of the search
    # used by test file and combined by '|'
    #
    [array]$addslideidscan = @('^_','.qptiff')
    [array]$addslideiddot = @('Full', 'Parameters')
    [array]$addslideid = @('^-')
    #
    samplereqs(){}
    samplereqs($mpath) : base($mpath){}
    samplereqs($mpath, $module) : base($mpath, $module){}
    samplereqs($mpath, $module, $slideid) : base($mpath, $module, $slideid){}
    samplereqs($mpath, $module, $batchid, $project) : base($mpath, $module, $batchid, $project){}
    #
    [switch]teststainfiles(){
        #
        $scans = $this.spathscans()
        #
        if ($scans -or (test-path $this.basepath)){
            return $true
        }
        #
        return $false
        #
    }
    #
    [switch]testscanfiles(){
        #
        if (test-path $this.basepath){
            return $true
        }
        #
        $scans = $this.spathscans()
        #
        if ($scans){
            foreach ($scan in $scans){
                $file = get-childitem ($scan.fullname, '*' -join '') -Include 'annotations.xml'
                if (get-content $file | Select-String "Acquired"){
                    return $true
                }
            }
        }
        #
        return $false 
        #
    }
    #
    [switch]testscanvalidationfiles(){
        #
        if (test-path $this.basepath){
            return $true
        }
        #
        $scans = $this.spathscans()
        #
        if ($scans){
            foreach ($scan in $scans){
                $file = get-childitem ($scan.fullname, '*' -join '') -Include 'BatchID.txt'
                if ($file){
                    return $true
                }
            }
        }
        #
        return $false 
        #
    }
    #
    [switch]testtransferfiles(){
        #
        $im3s = $this.countfiles(($this.Scanfolder() + '\MSI\*'), 'im3')
        #    
        if ($im3s -eq 0){
            return $false
        }
        #
        return $this.testfiles($this.scanfolder(), $this.transferreqfiles)
        #
    }
    #
    [switch]testshredxmlfiles(){
        return $this.testxmlfiles()
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
    [void]testim3mainfolder(){
        if (!(test-path $this.im3mainfolder())){
            Throw "im3 folder not found for:" + $this.im3mainfolder()
        }
    }
    #
    [switch]testmeanimagefiles(){
        #
        if ($this.vers -match '0.0.1'){
            #
            return $this.testfiles($this.im3mainfolder(), $this.meanimagereqfilesv1)
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
        <#
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
        #>
        return $true
    }
    #
    [switch]testbatchflatfieldfiles(){
        if ($this.moduleinfo.batchflatfield.vers -notmatch '0.0.1'){
            return $this.testpybatchflatfield()
        } else {
            return $this.testbatchflatfield()
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
        if ($this.teststatus){
            $this.ImportCorrectionModels($this.mpath, $false)
        } else{ 
            $this.ImportCorrectionModels($this.mpath)
        }
        #
        if ($this.corrmodels_data.slideid -notcontains $this.slideid){
            return $false
        }
        #
        if (!(test-path $this.pybatchflatfieldfullpath())){
            return $false
        }
        #
        return $true
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
        $p = ($this.meanimagefolder() + '\' +
         $this.slideid + '-mask_stack.bin')
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
        return $this.testfiles($this.warpoctetsfolder(),
         $this.warpoctetsreqfiles)
        #
    }
    #
    [switch]testbatchwarpkeysfiles(){
        #
        return $this.testfiles($this.warpbatchoctetsfolder(),
         $this.batchwarpkeysreqfiles)
        #
    }
    #
    [switch]testbatchwarpfitsfiles(){
        #
        return $this.testfiles($this.warpbatchfolder(),
         $this.batchwarpfitsreqfiles)
        #
    }
    #
    [switch]testimagecorrectionfiles(){
        #
        if (!$this.testfiles($this.flatwfolder(), 
                $this.im3constant, $this.imagecorrectionreqfiles[0])){
                    return $false
                }
        #
        if (!$this.testfiles($this.flatwfolder(), 
                $this.im3constant, $this.imagecorrectionreqfiles[1])){
                    return $false
                }
        #
        if (!$this.testfiles($this.flatwim3folder(), 
                $this.im3constant, $this.imagecorrectionreqfiles[2])){
                    return $false
                }
        #
        return $true
        #
    }
    #
    [switch]testmergefiles($cantibodies){
        #
        $this.getfiles('merge', $true)
        if (!$this.mergefiles){
            return $false 
        }
        #
        $date1 =  $this.getmindate('merge', $true)
        #
        foreach($antibody in $cantibodies){
            #
            $this.cantibody = $antibody
            $date2 = $this.getmaxdate('cantibody', $true)
            #
            if ($date2 -ge $date1){
                return $false
            }            
            #
        }
        #
        return $true
        #
    }
    #
    [switch]testimageqafiles(){
        #
        $this.getantibodies()
        #
        if ($this.checknewimageqa($this.antibodies)){
            return $false
        }
        #
        if(!$this.testimageqafile($this.antibodies)){
            return $false
        }
        #
        return $true
        #
    }
    #
    [switch]checknewimageqa($cantibodies){
        #
        $this.importimageqa($this.basepath, $cantibodies)
        #
        $data = $this.imageqa_data.slideid -contains $this.slideid
        #
        if (!$data){
            $this.addimageqa(
                $this.basepath, $this.slideid, $cantibodies)
            return $true
        }
        #
        return $false
        #
    }
    #
    [switch]testimageqafile($cantibodies){
        #
        $this.importimageqa($this.basepath, $cantibodies)
        #
        $data = $this.imageqa_data | 
            & { process {
                if ($_.SlideID -contains $this.slideid) { $_ }
            }}
        #
        foreach($antibody in $cantibodies){
            #
            if ($data.($antibody) -notcontains 'X'){
                return $false
            }         
            #
        }
        #
        return $true
        #
    }
    #
    [switch]testsegmapsfiles(){
        #
        return $this.testfiles($this.componentfolder(),
             $this.im3constant, $this.segmapsreqfiles)
        #
    }
    #
    [switch]testdbloadfiles(){
        return $true
    }
    <# --------------------------------------------
    testfiles
     test the files are accurate according 
     to the input. For input (2) checks if
     the files in the input ([filetype]req) array
     exist. For (3) input checks that the input 
     either exists or that the count is correct [and
     no files are empty (indicating a corrupt file).]
     --------------------------------------------
     Input 
        [string]path: path of files to test
        [string]source: the sample [filetype] 
        to compare n files with. 
        [array]testfiles: an array of files 
            to test from [filetype]req format
    --------------------------------------------
    Usage
        testfiles($path, [array]$testfiles)
        testfiles($path, $source, [array]$testfiles)
    -------------------------------------------- #>
    [switch]testfiles($path, [array]$testfiles){
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
    [switch]testfiles($path, $source, [array]$testfiles){
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
    <# --------------------------------------------
    testfile 
     test if the file exists. If the file starts with 
     '-' prepend slideid, if the file is Parametes\ Full
     then prepend '.'slideid. If the file has
     '_' or matches qptiff add 
    -------------------------------------------- #>

    [switch]testfile($path, $file){
        #
        if ($file -match ($this.addslideid -join '|')){
            $file = $this.slideid + $file
        }
        #
        if ($file -match ($this.addslideidscan -join '|')){
            $file = $this.slideid + '_' + $this.Scan() +  $file
        }
        #
        if ($file -match ($this.addslideiddot -join '|')){
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
        $count1 = $this.countfiles(
            $this.($source + 'folder')(), $this.($source + 'constant')
        )
        #
        $count2 = $this.countfiles(
            $this.($testfile + 'folder')(), $this.($testfile + 'constant')
        )
        #
        if ($count1 -ne $count2){
            return $false
        }
        #
        return $true
        #
    }
 }
 #