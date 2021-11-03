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
    #
    sampledef(){}
    #
    # sampledef([string]$mpath) : base($mpath){}
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
        $this.DefRoot($mpath)
    }
    #
    Batch(
        [string]$batchid="",
        [string]$mpath,
        [PSCustomObject]$slides
    ){
        $this.ParseAPIDdefbatch($batchid, $slides)
        $this.DefRoot($mpath)
    }
    #
    [void]ParseAPIDdef([string]$slideid, [PSCustomObject]$slides){
        $slide = $slides | 
                Where-Object -FilterScript {$_.SlideID -eq $slideid.trim()}
        #
        if (!$slide){
            Throw 'Not a valid slideid'
        }
        $this.slideid = $slide.SlideID.trim()
        $this.project = $slide.Project
        $this.cohort = $slide.Cohort
        #
        if ($slide.BatchID.Length -eq 1){
            $this.BatchID = '0' + $slide.BatchID
        } else {
            $this.BatchID = $slide.BatchID 
        }
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
        }
        $this.project = $batch.Project[1]
        $this.cohort = $batch.Cohort[1]
        #
        if ($batch.BatchID[1].Length -eq 1){
            $this.BatchID = '0' + $batch.BatchID[1]
        } else {
            $this.BatchID = $batch.BatchID[1] 
        }
        $this.slideid = $this.BatchID
        #
        $this.batchslides = $batch
        #
    }
    #
    [void]DefRoot([string]$mpath){
        $this.mpath = $mpath
        $project_dat = $this.importcohortsinfo($this.mpath)
        $project_dat = $project_dat | 
                Where-Object -FilterScript {$_.Project -eq $this.project}
        $this.basepath = '\\' + $project_dat.dpath + '\' + $project_dat.dname
        $this.project_data = $project_dat
    }
    #
    [string]im3folder(){
        $path = $this.basepath + '\' + $this.slideid + '\im3'
        return $path

    }
    #
    [string]Scan(){
        $path = $this.basepath + '\' + $this.slideid + '\im3\Scan*'
        $paths = gci $path
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
    [switch]testxmlfiles(){
        #
        $xml = $this.xmlfolder()
        $im3s = (gci ($this.Scanfolder() + '\MSI\*') *im3).Count + 2
        #
        if (!(test-path $xml)){
            return $false
        }
        #
        # check files = im3s
        #
        $files = (gci ($xml + '\*') '*xml').Count
        if (!($im3s -eq $files)){
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
        }
        #
        return $true
        #
    }
    #
    [switch]testimagecorrectionfiles(){
        #
        $im3s = (gci ($this.Scanfolder() + '\MSI\*') *im3).Count
        #
        $paths = @($this.flatwim3folder(), ('\\'+$this.flatwfolder()), ('\\'+$this.flatwfolder()))
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
            $files = (gci ($paths[$i] + '\*') $filetypes[$i]).Count
            if (!($im3s -eq $files)){
                return $false
            }
        }
        #
        return $true
        #
    }
    #
}
