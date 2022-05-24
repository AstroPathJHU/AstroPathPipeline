<# -------------------------------------------
 samplefiles
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to define find or create specified 
 files or folders used throughout the module. 
     Here we define some of the variables for 
     for files we may want to track as part of the pipeline define:
     - a [filetype] in the [filetypes] array that will server as a 
     description of the files. 
     - a [filetype]files [system.object] to store the files object
     recieved from get-childitem. 
     - a [filetype]constant that will define the search pattern when 
     searching for the files.
     - a [filetype]folder() method that will define the file location
     for the files.
 -------------------------------------------#>
 class samplefiles : sampledef {
    #
    [array]$filetype = @('im3','fw','fw01','raw','flatwim3',
        'xml','exposurexml','algorithm','project','segmap','merge',
        'cellseg','binseg','cellsegsum','component','cantibody')
    #
    [system.object]$im3files
    [system.object]$fwfiles
    [system.object]$fw01files
    [system.object]$rawfiles
    [system.object]$flatwim3files
    [system.object]$xmlfiles
    [system.object]$exposurexmlfiles
    [system.object]$algorithmfiles
    [system.object]$projectfiles
    [system.object]$segmapfiles
    [system.object]$mergefiles
    [system.object]$cellsegfiles
    [system.object]$binsegfiles
    [system.object]$cellsegsumfiles
    [system.object]$componentfiles
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
    [string]$cellsegsumconstant = '_cell_seg_data_summary.txt'
    [string]$componentconstant = '_component_data.tif'
    [string]$cantibodyconstant = '_cell_seg_data.txt'
    #
    samplefiles(){}
    samplefiles($mpath) : base($mpath){}
    samplefiles($mpath, $module) : base($mpath, $module){}
    samplefiles($mpath, $module, $slideid) : base($mpath, $module, $slideid){}
    samplefiles($mpath, $module, $batchid, $project) : base($mpath, $module, $batchid, $project){}
    #
    [string]spath(){
        return ($this.project_data.spath, $this.slideid -join '\')
    }
    #
    [PSCustomObject]spathscans(){
        if (test-path $this.spath()){
            return (get-childitem ($this.spath(), 'Scan*' -join '\'))
        } else {
            return @()
        }
    }
    #
    [string]im3mainfolder(){
        return ($this.basepath + '\' + $this.slideid + '\im3')
    }
    #
    [string]upkeepfolder(){
        return ($this.basepath + '\upkeep_and_progress')
    }
    #
    [string]Scan(){
        $paths = get-childitem ($this.im3mainfolder(), 'Scan*' -join '\')
        return ($paths | 
            select-object *, @{n = "IntVal"; e = {[int]$_.Name.substring(4)}} |
            sort-object IntVal |
            Select-Object -Last 1).name
    }
    #
    [string]Scanfolder(){
        $path = $this.basepath + '\' + $this.slideid + 
            '\im3\'+$this.Scan()
        return $path

    }
    #
    [string]batchfolder(){
        $path = $this.basepath + '\Batch'
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
    [string]batchwarpingfile(){
        $path = $this.basepath +'\warping\warping_BatchID_' + 
            $this.BatchID + '.csv'
        return $path
    }
    #
    [string]pybatchflatfield(){
        $this.ImportCorrectionModels($this.mpath)
        if ($this.slideid -notcontains $this.batchid){
            $file = ($this.corrmodels_data | & { process {
                if ( $_.slideid -contains $this.slideid) {$_}
            }}).FlatfieldVersion
        } else  {
            $file1 = ($this.corrmodels_data | & { process { 
                if (
                    $_.BatchID.padleft(2, '0') -contains $this.batchid.padleft(2, '0') -and
                    $_.Project.padleft(3,'0') -contains $this.project.padleft(3,'0') 
                ) { $_ }
            }}).FlatfieldVersion
           if ($file1.Count -gt 1){
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
          return (
            $this.mpath + '\flatfield\flatfield_' +
                $this.pybatchflatfield() + '.bin'
          )
    }
    #
    [string]CheckSumsfile(){
        return ($this.Scanfolder() + '\CheckSums.txt')
    }
    #
    [string]im3folder(){
        return ($this.Scanfolder() + '\MSI')
    }
    #
    [string]informfolder(){
        return (
            $this.basepath, $this.slideid, 'inform_data' -join '\'
        )

    }
    #
    [string]componentfolder(){
        return (
            $this.basepath, $this.slideid, 
                'inform_data\Component_Tiffs' -join '\'
        )
    }
    #
    [string]segmapfolder(){
        return (
            $this.basepath, $this.slideid, 
                'inform_data\Component_Tiffs' -join '\'
        )
    }
    #
    [string]phenotypefolder(){
        return (
            $this.basepath, $this.slideid, 
                'inform_data\Phenotyped' -join '\'
        )

    }
    #
    [string]informantibodylogfile($antibody){
        return (
            $this.phenotypefolder(), $antibody, 'batch.log' -join '\'
        )
    }
    #
    [string]cantibodyfolder(){
        return (
            $this.phenotypefolder(), $this.cantibody -join '\'
        )
    }
    #
    [string]mergefolder(){
        return (
            $this.basepath, $this.slideid, 
                'inform_data\Phenotyped\Results\Tables' -join '\'
        )

    }
    #
    [string]mergealgorithmsfile(){
        return (
            $this.basepath, $this.slideid, 
                'inform_data\Phenotyped\Results\algorithms.txt' -join '\'
        )

    }
    #
    [string]xmlfolder(){
        return (
            $this.basepath, $this.slideid, 'im3\xml' -join '\'
        )

    }
    #
    [string]exposurexmlfolder(){
        return $this.xmlfolder()
    }
    #
    [string]meanimagefile(){
        return (
            $this.basepath, $this.slideid, 
                'im3', ($this.slideid + '-mean.flt') -join '\'
        )
    }
    #
    [string]meanimagefolder(){
        return ($this.im3mainfolder() + '\meanimage')
    }
    #
    [string]flatwim3folder(){
        return (
            $this.basepath, $this.slideid, 'im3\flatw' -join '\'
        )
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
        return (
            $this.basepath + '\Batch\MergeConfig_' + 
                $this.BatchID
        )
    }
    #
    [string]warpoctetsfolder(){
        return (
            $this.basepath, $this.slideid,
                'im3\warping\octets' -join '\'
        )
    }
    #
    [string]warpoctetsfile(){
        return (
            $this.warpoctetsfolder(), '\', $this.slideid, 
                '-all_overlap_octets.csv' -join ''
        )
    }
    #
    [string]warpbatchfolder(){
        return ($this.basepath +'\warping\Batch_' + $this.BatchID)
    }
    #
    [string]warpprojectfolder(){
        return ($this.mpath +'\warping\Project_' + $this.project)
    }
    #
    [string]warpbatchoctetsfolder(){
        return (
            $this.basepath +'\warping\Batch_' +
                $this.BatchID + '\octets'
        )
    }
    #
    [string]warpprojectoctetsfolder(){
        return (
            $this.basepath + '\warping\Project_' +
                $this.project + '\octets'
        )
    }
    #
    [array]getalgorithms(){
        #
        $this.findantibodies() 
        return (
            $this.antibodies | 
            foreach-object {
                $this.getalgorithms($_)
            }
        )
        #
    }
    #
    [string]getalgorithms($cantibody){
        #
        return ((get-content $this.informantibodylogfile($cantibody) | 
            Where-Object {$_ -match 'algorithm'}) -split 'algorithm')[1].trim()
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
    [datetime]getmindate($source, $forceupdate){
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
        if (!$date){
            $date = get-date
        }
        #
        return $date
        #
    }
    #
    [datetime]getmaxdate($source, $forceupdate){
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
        if (!$date){
            $date = get-date -Date "6/25/2019 12:30:22"
        }
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
            $names = $this.('get' + $type)(
                $this.($source + 'folder')(), $this.($source + 'constant')
            )
        } else {
            $names = ($this.getfiles(
                $source)).($type)
        }
        #
        return $names
        #
    }
    #
    [switch]checkforfiles($source, $filetype){
        #
        if (test-path ($source, '*' -join '\') -include ('*' + $filetype)){
            return $true
        } 
        #
        return $false
        #
    }
    #
    [switch]checkforfiles($filetype){
        return $this.checkforfiles(
            $this.($filetype + 'folder')(),
            $this.($filetype + 'constant')
        )
    }
    #
    [switch]checkforcontents($source, $filetype, $contents){
        if (
            get-childitem ($source, '*' -join '\') `
                -Include ('*' + $filetype) | 
                get-content | 
                select-string $contents
        ){
            return $true
        }
        return $false
    }
    #
    [switch]checkforcontents($filetype){
        return (
            $this.checkforcontents(
                $this.($filetype + 'folder')(),
                $this.($filetype + 'constant'),
                $this.($filetype + 'contents')
            )
        )
    }
 }