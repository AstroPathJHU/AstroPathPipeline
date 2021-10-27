
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
    sampledef($mpath, $module, $slideid, $project){
        $this.mpath = $mpath
        $this.module = $module 
        $this.sampledefbatch($slideid, $project)
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
        $slide = $slides | `
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
        $batch = $slides | `
                Where-Object -FilterScript {$_.BatchID -eq $mbatchid.trim() -and $_.Project -eq $this.project.trim()}
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
        $project_dat = $project_dat | `
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
        $scan = $paths | select-object *, @{n = "IntVal"; e = {[int]$_.Name.substring(4)}} | sort-object IntVal | Select-Object -Last 1
        return $scan.Name
    }
    #
    [string]Scanfolder(){
        $path = $this.basepath + '\' + $this.slideid + '\im3\'+$this.Scan()
        return $path

    }
    #
    [string]qptifffile(){
        $path = $this.Scanfolder() + '\' + $this.slideid + '_' + $this.Scan() + '.qptiff'
        return $path
    }
    #
    [string]annotationxml(){
        $path = $this.Scanfolder() + '\' + $this.slideid + '_' + $this.Scan() + '_annotations.xml'
        return $path
    }
    #
    [string]batchIDfile(){
        $path = $this.Scanfolder() + '\BatchID.txt'
        return $path
    }
    #
    [string]batchflatfield(){
        $path = $this.basepath +'\flatfield\flatfield_BatchID_' + $this.BatchID + '.bin'
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
        $path = $this.basepath + '\' + $this.slideid + '\inform_data'
        return $path

    }
    #
    [string]componentfolder(){
        $path = $this.basepath + '\' + $this.slideid + '\inform_data\Component_Tiffs'
        return $path
    }
    #
    [string]phenotypefolder(){
        $path = $this.basepath + '\' + $this.slideid + '\inform_data\Phenotyped'
        return $path

    }
    #
    [string]xmlfolder(){
        $path = $this.basepath + '\' + $this.slideid + '\im3\xml'
        return $path

    }
    #
    [string]meanimagefile(){
        $path = $this.basepath + '\' + $this.slideid + '\im3\' + $this.slideid + '-mean.flt'
        return $path
    }
    #
    [string]meanimagefolder(){
        $path = $this.im3folder() + '\meanimage'
        return $path
    }
    #
    [string]flatwim3folder(){
        $path = $this.basepath + '\' + $this.slideid + '\im3\flatw'
        return $path
    }
    #
    [string]flatwfolder(){
        $path = '\\'+$this.project_data.fwpath + '\' + $this.slideid
        return $path
    }
    #
    [string]mergeconfigfile(){
        $path = $this.basepath + '\Batch\MergeConfig_' + $this.BatchID
        return $path
    }
    #
    [void]testim3folder(){
        if (!(test-path $this.im3folder())){
            Throw "im3 folder not found for:" + $this.im3folder()
        }
    }
    #
    [void]testbatchflatfield(){
        if (!(test-path $this.batchflatfield())){
            Throw "batch flatfield not found for:" + $this.batchflatfield()
        }
    }

}
