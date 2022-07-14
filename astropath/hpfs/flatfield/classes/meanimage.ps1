<#
--------------------------------------------------------
meanimage
Benjamin Green, Andrew Jorquera
Last Edit: 10/19/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[hashtable]: the 3 part array of project, slideid, process loc
    E.g. @('7','M18_1','\\bki08\e$')
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [meanimage]::new($task, $sample)
       $a.runmeanimage()
--------------------------------------------------------
#>
Class meanimage : moduletools {
    #
    [string]$pytype = 'sample'
    #
    meanimage([hashtable]$task, [launchmodule]$sample) : base ([hashtable]$task, [launchmodule]$sample){
        $this.funclocation = '"' + $PSScriptRoot + '\..\funcs"'  
        $this.flevel = [FileDownloads]::IM3 + [FileDownloads]::XML
    }
    <# -----------------------------------------
     RunMeanImage
     Run mean image
     ------------------------------------------
     Usage: $this.RunMeanImage()
    ----------------------------------------- #>
    [void]RunMeanImage(){
        $this.cleanupbase()
        $this.sample.CreateNewDirs($this.processloc)
        $this.checkapiddef()
        $this.checksampledef()
        $this.fixSIDs()
        $this.fixmlids()
        $this.DownloadFiles()
        $this.ShredDat()
        $this.GetMeanImage()
        $this.returndata()
        $this.cleanup()
        $this.datavalidation()
    }
   <# -----------------------------------------
     GetMeanImage
        Get the mean image
     ------------------------------------------
     Usage: $this.GetMeanImage()
    ----------------------------------------- #>
    [void]GetMeanImage(){
        #
        if ($this.vers -match '0.0.1'){
            $this.GetMeanImageMatlab()
        }
        else{
            $this.GetMeanImagePy()
        }
        #
    }
    <# -----------------------------------------
     GetMeanImageMatlab
        Get mean image using matlab code
     ------------------------------------------
     Usage: $this.GetMeanImageMatlab()
    ----------------------------------------- #>
    [void]GetMeanImageMatlab(){
        $this.sample.info("started mean image sample -- matlab")
        $taskname = 'raw2mean'
        $matlabtask = ";raw2mean('" + $this.processvars[1] + 
            "', '" + $this.sample.slideid + "');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask)
        $this.sample.info("finished mean image sample -- matlab")
    }
    <# -----------------------------------------
     GetMeanImagePy
        Get mean image using python code
     ------------------------------------------
     Usage: $this.GetMeanImagePy()
    ----------------------------------------- #>
    [void]GetMeanImagePy(){
        $this.sample.info("started mean image sample -- python")
        $this.getmodulename()
        $taskname = $this.pythonmodulename
        #
        $dpath = $this.sample.basepath + ' '
        $rpath = $this.processvars[1]
        $pythontask = $this.('getpythontask' + $this.pytype)($dpath, $rpath)
        #
        $this.runpythontask($taskname, $pythontask)
        $this.sample.info("finished mean image sample -- python")
    }
    #
    [string]getpythontasksample($dpath, $rpath){
        #
        $globalargs = $this.buildpyopts()
        $pythontask = ($this.pythonmodulename,
            $dpath, 
            $this.sample.slideid,
            '--shardedim3root', $rpath, 
            ' --workingdir', ($this.processvars[0] + '\meanimage'), 
            "--njobs '8'",
            $globalargs -join ' ')
        #
        return $pythontask
    }
    #
    [string]getpythontaskcohort($dpath, $rpath){
        #
        $globalargs = $this.buildpyopts('cohort')
        $pythontask = ($this.pythonmodulename,
             $dpath, 
             '--sampleregex', $this.sample.slideid,
             '--shardedim3root', $rpath, 
             ' --workingdir', ($this.processvars[0] + '\meanimage'), 
            "--njobs '8'",
            $globalargs -join ' ')
        #
        return $pythontask
    }
    #
    [void]getmodulename(){
        $this.pythonmodulename = ('meanimage', $this.pytype -join '')
    }
    <# -----------------------------------------
     returndata
        return data
     ------------------------------------------
     Usage: $this.returndata()
    ----------------------------------------- #>
    [void]returndata(){
        if (!$this.processvars[4]){
            return
        }
        $this.sample.info("return data started")
        if ($this.vers -match '0.0.1'){
            $this.ReturnDataMatlab()
        }
        else{
            $this.ReturnDataPy()
        }
        $this.sample.info("return data finished")
    }
    <# -----------------------------------------
     ReturnDataMatlab
     returns data to source path
     ------------------------------------------
     Usage: $this.ReturnDataMatlab()
    ----------------------------------------- #>
    [void]ReturnDataMatlab(){
        #
		$des = $this.sample.im3mainfolder()
        #
        $sor = $this.processvars[1] + '\flat\' + 
            $this.sample.slideid + '\*.flt'
        #
        if (!(Get-ChildItem $sor)){
            Throw 'no .flt file found, matlab meanimage failed'
        }                        
        $this.sample.copy($sor, $des) 
        #
        $sor = $sor -replace 'flt', 'csv'
        if (!(Get-ChildItem $sor)){
            Throw 'no .csv file found, matlab meanimage failed'
        }
        $this.sample.copy($sor, $des) 
        #
    }
    <# -----------------------------------------
     ReturnDataPy
     returns data to im3 folder
     ------------------------------------------
     Usage: $this.ReturnDataPy()
    ----------------------------------------- #>
    [void]ReturnDataPy(){
        if ($this.processvars[4]){
            #
		    $des = $this.sample.im3mainfolder() + '\meanimage'
            $sor = $this.processvars[0] +'\meanimage'
            #
            $this.sample.copy($sor, $des, '*', 30)
            #
        }
    }
    <# -----------------------------------------
     cleanup
     cleanup the data directory
     ------------------------------------------
     Usage: $this.cleanup()
    ----------------------------------------- #>
    [void]cleanup(){
        #
        $this.sample.info("cleanup started")
        $this.silentcleanup()
        $this.sample.info("cleanup finished")
        #
    }
    <# -----------------------------------------
     silentcleanup
     silentcleanup
     ------------------------------------------
     Usage: $this.silentcleanup()
    ----------------------------------------- #>
    [void]silentcleanup(){
        #
        if ($this.processvars[4]){
            $this.sample.removedir($this.processloc)
        }
        #
    }
    <# -----------------------------------------
     cleanupbase
     remove old results
     ------------------------------------------
     Usage: $this.cleanupbase()
    ----------------------------------------- #>
    [void]cleanupbase(){
        #
        $this.sample.removedir($this.sample.meanimagefolder())
        #
    }
    <# -----------------------------------------
     datavalidation
     Validation of output data
     ------------------------------------------
     Usage: $this.datavalidation()
    ----------------------------------------- #>
    [void]datavalidation(){
        if (!$this.sample.testmeanimagefiles()){
            throw 'Output files are not correct'
        }
    }
}
