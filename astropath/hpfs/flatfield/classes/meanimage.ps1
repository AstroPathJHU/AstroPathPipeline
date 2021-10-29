<#
--------------------------------------------------------
meanimage
Created By: Andrew Jorquera
Last Edit: 10/19/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[array]: the 3 part array of project, slideid, process loc
    E.g. @('7','M18_1','\\bki08\e$')
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [meanimage]::new($task, $sample)
       $a.runmeanimage()
--------------------------------------------------------
#>
Class meanimage : moduletools {
    #
    meanimage([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.flevel = [FileDownloads]::IM3 + [FileDownloads]::XML
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'  
    }
    <# -----------------------------------------
     RunMeanImage
     Run mean image
     ------------------------------------------
     Usage: $this.RunMeanImage()
    ----------------------------------------- #>
    [void]RunMeanImage(){
        $this.DownloadFiles()
        $this.ShredDat()
        $this.GetMeanImage()
        $this.returndata()
        $this.cleanup()
    }
   <# -----------------------------------------
     GetMeanImage
        Get the mean image
     ------------------------------------------
     Usage: $this.GetMeanImage()
    ----------------------------------------- #>
    [void]GetMeanImage(){
        if ($this.vers -eq '0.0.1'){
            $this.GetMeanImageMatlab()
        }
        else{
            $this.GetMeanImagePy()
        }
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
        $matlabtask = ";raw2mean('"+$this.processvars[1]+"', '"+$this.sample.slideid+"');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
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
        $taskname = 'meanimagesample'
        $dpath = $this.sample.basepath + ' '
        # $dpath = '\\bki04\Clinical_Specimen '
        $rpath = $this.processvars[1]
        $pythontask = 'meanimagesample ' + $dpath + $this.sample.SlideID + 
         ' --shardedim3root ' + $rpath +
         ' --workingdir ' + $this.processvars[0] + '\meanimage' +
         " --njobs '8' --allow-local-edits --skip-start-finish"
        $this.runpythontask($taskname, $pythontask)
        $this.sample.info("finished mean image sample -- python")
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
        if ($this.vers -eq '0.0.1'){
            $this.ReturnDataMatlab()
        }
        else{
            $this.ReturnDataPy()
        }
    }
    <# -----------------------------------------
     ReturnDataMatlab
     returns data to source path
     ------------------------------------------
     Usage: $this.ReturnDataMatlab()
    ----------------------------------------- #>
    [void]ReturnDataMatlab(){
        #
		$des = $this.sample.im3folder()
        #
        $sor = $this.processvars[1] +'\flat\'+$this.sample.slideid+'\*.flt'
        xcopy $sor, $des /q /y /z /j /v | Out-Null
        #
        $sor = $this.processvars[1] + '\flat\'+$this.sample.slideid+'\*.csv'
        xcopy $sor, $des /q /y /z /j /v | Out-Null
        #
        $sor = $this.processvars[1] + '\flat\'+$this.sample.slideid
        Remove-Item $sor -force -recurse
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
		    $des = $this.sample.im3folder() + '\meanimage'
            $sor = $this.processvars[0] +'\meanimage'
            xcopy $sor, $des /q /y /z /j /v /s /i | Out-Null
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
        if ($this.processvars[4]){
            Get-ChildItem -Path $this.processloc -Recurse | Remove-Item -force -recurse
            Remove-Item $this.processloc -Force
        }
        $this.sample.info("cleanup finished")
        #
    }
}
