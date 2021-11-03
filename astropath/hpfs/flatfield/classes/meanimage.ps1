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
    meanimage([array]$task, [launchmodule]$sample) : base ([array]$task, [launchmodule]$sample){
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
        $this.checkexternalerrors()
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
        $rpath = $this.processvars[1]
        $pythontask = 'meanimagesample ' + $dpath + $this.sample.SlideID + 
         ' --shardedim3root ' + $rpath +
         ' --workingdir ' + $this.processvars[0] + '\meanimage' +
         " --njobs '8' --allow-local-edits --skip-start-finish"
        $this.runpythontask($taskname, $pythontask)
        $this.checkexternalerrors()
        $this.sample.info("finished mean image sample -- python")
    }
    <# -----------------------------------------
     checkexternalerrors
        checkexternalerrors
     ------------------------------------------
     Usage: $this.checkexternalerrors()
    ----------------------------------------- #>
    [void]checkexternalerrors(){
        #
        if ($this.logoutput){
            if ($this.processvars[4]){
                $this.sample.removedir($this.processloc)
            }
            Throw (($this.logoutput.trim() -ne '') -notmatch 'ERROR')
        }
        #
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
        if ($this.vers -match '0.0.1'){
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
        $sor = $this.processvars[1] + '\flat\' + 
            $this.sample.slideid + '\*.flt'
        #
        if (!(gci $sor)){
            Throw 'no .flt file found, matlab meanimage failed'
        }                        
        $this.sample.copy($sor, $des) 
        #
        $sor = $sor -replace 'flt', 'csv'
        if (!(gci $sor)){
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
		    $des = $this.sample.im3folder() + '\meanimage'
            $sor = $this.processvars[0] +'\meanimage'
            $this.sample($sor, $des, '*', 30)
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
            $this.sample.removedir($this.processloc)
        }
        $this.sample.info("cleanup finished")
        #
    }
}
