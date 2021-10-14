<#
--------------------------------------------------------
meanimage
Created By: Andrew Jorquera
Last Edit: 09/29/2021
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
        $this.cleanup()
    }
   <# -----------------------------------------
     GetMeanImage
        Get the mean image
     ------------------------------------------
     Usage: $this.GetMeanImage()
    ----------------------------------------- #>
    [void]GetMeanImage(){
        $this.sample.info("started getting mean image")
        if ($this.vers -eq '0.0.1'){
            $this.RunMeanImageMatlab()
            $this.returndata()
        }
        else{
            $this.RunMeanImagePy()
        }
        $this.sample.info("finished getting mean image")
    }
    <# -----------------------------------------
     RunMeanImageMatlab
        Run mean image matlab code
     ------------------------------------------
     Usage: $this.RunMeanImageMatlab()
    ----------------------------------------- #>
    [void]RunMeanImageMatlab(){
        $taskname = 'raw2mean'
        $matlabtask = ";raw2mean('"+$this.processvars[1]+"', '"+$this.sample.slideid+"');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask, $this.funclocation)
    }
    <# -----------------------------------------
     RunMeanImagePy
        Run mean image python code
     ------------------------------------------
     Usage: $this.RunMeanImagePy()
    ----------------------------------------- #>
    [void]RunMeanImagePy(){
        $this.sample.info("started mean image sample python script")
        $taskname = 'meanimagesample'
        $dpath = '\\bki04\Clinical_Specimen '
        $rpath = $this.processloc + '\flatw\ '
        $pythontask = 'meanimagesample ' + $dpath + $rpath + $this.sample.SlideID + ' --workingdir ' + $this.processloc + '\meanimage'+ " --njobs '8' --allow-local-edits"
        $this.runpythontask($taskname, $pythontask)
        $this.sample.info("finished mean image sample python script")
    }
    <# -----------------------------------------
     RunMeanImageComparison
        Run mean image comparison python code
     ------------------------------------------
     Usage: $this.RunMeanImageComparison()
    ----------------------------------------- #>
    [void]RunMeanImagePy(){
        $this.sample.info("started mean image sample python script")
        $taskname = 'meanimagesample'
        $dpath = '\\bki04\Clinical_Specimen '
        $rpath = $this.processvars[1] + ' '
        $pythontask = 'meanimagesample ' + $dpath + $rpath + $this.sample.SlideID + 
         ' --workingdir ' + $this.processvars[0] + '\meanimage'+ " --njobs '8' --allow-local-edits"
        $this.runpythontask($taskname, $pythontask)
        $this.sample.info("finished mean image sample python script")
    }
    <# -----------------------------------------
     returndata
     returns data to source path
     ------------------------------------------
     Usage: $this.returndata()
    ----------------------------------------- #>
    [void]returndata(){
        #
        $this.sample.info("Return data started")
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
        $this.sample.info("Return data finished")
        #
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
        #
        if ($this.processvars[4]){
            Get-ChildItem -Path $this.processloc -Recurse | Remove-Item -force -recurse
        }
        $this.sample.info("cleanup finished")
        #
    }
}
