<#
--------------------------------------------------------
meanimage
Created By: Andrew Jorquera -JHU
Last Edit: 09/08/2021
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
Class meanimage {
    #
    [string]$processloc
    [launchmodule]$sample
    [array]$processvars
    [string]$convertim3pathlogshred
    [string]$meanimagelog
    [int]$downloadim3ii = 0
    [string]$vers
    [string]$funclocation = '"'+$PSScriptRoot + '\..\funcs"'

    #
    meanimage([array]$task,[launchmodule]$sample){
        #
        $this.sample = $sample
        $this.BuildProcessLocPaths($task)
        $this.vers = $this.sample.GetVersion($this.sample.mpath, $this.sample.module, $task[0])
        #
    }
    <# -----------------------------------------
     BuildProcessLocPath
     build the processing specimens directory paths
     if it does not exist. If input is '*' then
     work 'in place'
     ------------------------------------------
     Usage: $this.BuildProcessLoc()
    ----------------------------------------- #>
    [void]BuildProcessLocPaths($task){
        #
        $fwpath = '\\'+$this.sample.project_data.fwpath
        $this.processvars = @($this.sample.basepath, $fwpath)
        #
        # If processloc is not '*' a processing destination was added as input, correct the paths to analyze from there
        #
        if ($task[2]){
             $this.processloc = ($task[2]+'\processing_meanimage\'+$task[1])
             #
             $this.convertim3pathlogshred = $this.processloc + '\convertIM3pathshred.log'
             $this.meanimagelog = $this.processloc + '\meanimage.log'
             #
             $this.processvars[0] = $this.processvars[0] -replace [regex]::escape($this.sample.basepath), $this.processloc
             $this.processvars[1] = $this.processvars[1] -replace [regex]::escape('\\'+$this.sample.project_data.fwpath), ($this.processloc+'\flatw')
             $this.downloadim3ii = 1
        } else {
             $this.convertim3pathlogshred = $this.sample.im3folder() + '\convertIM3pathshred.log'
             $this.meanimagelog = $this.sample.im3folder() + '\meanimage.log'
        }
        #
    }
    <# -----------------------------------------
     RunMeanImage
     Run mean image
     ------------------------------------------
     Usage: $this.RunMeanImage()
    ----------------------------------------- #>
    [void]RunMeanImage(){
        $this.DownloadIm3()
        $this.ShredDat()
        $this.GetMeanImage()
        $this.returndata()
        $this.cleanup()
    }
    #
    <# -----------------------------------------
     DownloadIm3
     Download the im3s to process; reduces network
     strain and frequent network errors while 
     processing
     ------------------------------------------
     Usage: $this.DownloadIm3()
    ----------------------------------------- #>
    [void]DownloadIm3(){
        #
        if ($this.downloadim3ii -eq 1){
            $this.sample.info("Download im3s started")
            foreach($ii in @(0,1)){
                if (test-path $this.processvars[$ii]){
                        remove-item $this.processvars[$ii] -force -Recurse -EA STOP
                    }
                New-Item $this.processvars[$ii] -itemtype "directory" -EA STOP | Out-NULL
            }
            # im3s
            $des = $this.processvars[0] +'\'+$this.sample.slideid+'\im3\'+$this.sample.Scan()+,'\MSI'
            $sor = $this.sample.MSIfolder()
            robocopy $sor $des *im3 -r:3 -w:3 -np -mt:30 |out-null
            if(!(((gci ($sor+'\*') -Include '*im3').Count) -eq (gci $des).count)){
                Throw 'im3s did not download correctly'
            }
            $this.sample.info("Download im3s finished")
            #
        }
    }
    <# -----------------------------------------
     ShredDat
        Extract data.dat and xml files
     ------------------------------------------
     Usage: $this.ShredDat()
    ----------------------------------------- #>
    [void]ShredDat(){
        $this.sample.info("Shred data started")
        ConvertIM3Path $this.processvars[0] $this.processvars[1] $this.sample.slideid -s -verbose 4>&1 >> $this.convertim3pathlogshred
        $log = $this.sample.GetContent($this.convertim3pathlogshred) |
             where-object {$_ -notlike '.*' -and $_ -notlike '*PM*' -and $_ -notlike '*AM*'} | 
             foreach {$_.trim()}
        $this.sample.info($log)
        remove-item $this.convertim3pathlogshred -force -ea Continue
        $this.sample.info("Shred data finished")
    }
   <# -----------------------------------------
     GetMeanImage
        Get the mean image
     ------------------------------------------
     Usage: $this.GetMeanImage()
    ----------------------------------------- #>
    [void]GetMeanImage(){
        $this.sample.info("started getting mean image")
        $applytask = ";raw2mean('"+$this.processvars[1]+"', '"+$this.sample.slideid+"');exit(0);"
        matlab -nosplash -nodesktop -minimize -sd $this.funclocation -r $applytask -wait >> $this.meanimagelog
        if (test-path $this.meanimagelog){
            remove-item $this.meanimagelog -force -ea Continue
        }
        $this.sample.info("finished getting mean image")
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
        if ($this.downloadim3ii -eq 1){
            # flt and csv
            $sor = $this.processvars[1] +'\flat'
            $des = $this.sample.im3folder()
            robocopy $sor $des *.flt -r:3 -w:3 -np -mt:30 |out-null
            robocopy $sor $des *.csv -r:3 -w:3 -np -mt:30 |out-null
            #
        }
        #
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
        #if ($this.downloadim3ii -eq 1) {
        Get-ChildItem -Path $this.processloc -Recurse | Remove-Item -force -recurse
        #}
        #
        $this.sample.info("cleanup finished")
        #
    }
}
