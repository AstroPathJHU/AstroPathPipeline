class sampledef : sharedtools{
    [string]$slideid
    [string]$project
    [string]$cohort
    [string]$BatchID
    [string]$basepath
    [string]$project_data
    #
    sampledef(){}
    #
    sampledef([string]$slideid
        ){
            $this.slideid = $slideid 
        }

    #
    sampledef(
        [string]$slideid,
        [string]$mpath
    ){
        #
        $slides = $this.importslideids($mpath)
        $this.Sample($slideid, $mpath, $slides)
        #
    }
    #
     sampledef(
        [string]$slideid="",
        [string]$mpath,
        [PSCustomObject]$slides
    ){
        $this.Sample($slideid, $mpath, $slides)
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
    [void]ParseAPIDdef([string]$slideid, [PSCustomObject]$slides){
        #
        $slide = $slides | `
                Where-Object -FilterScript {$_.SlideID -eq $slideid.trim()}
        #
        $this.slideid = $slide.SlideID.trim()
        $this.project = $slide.Project
        $this.cohort = $slide.Cohort
        $this.batchid = $slide.BatchID
    }
    #
    [void]DefRoot([string]$mpath){
        $this.mpath = $mpath
        $project_dat = $this.importcohortsinfo($this.mpath)
        $project_dat = $project_dat[1] | `
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
    [string]Scanfolder(){
        $path = $this.basepath + '\' + $this.slideid + '\im3\Scan*'
        $paths = gci $path
        $scan = $paths | select-object *, @{n = "IntVal"; e = {[int]$_.Name.substring(4)}} | sort-object IntVal | Select-Object -Last 1
        $path = $Scan.FullName
        return $path

    }
    #
    [string]informfolder(){
        $path = $this.basepath + '\' + $this.slideid + '\inform_data'
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
    [string]flatwim3folder(){
        $path = $this.basepath + '\' + $this.slideid + '\im3\flatw'
        return $path
    }
    #
    [string]flatwfolder(){
        $path = $this.project_data + '\' + $this.slideid
        return $path
    }

}
