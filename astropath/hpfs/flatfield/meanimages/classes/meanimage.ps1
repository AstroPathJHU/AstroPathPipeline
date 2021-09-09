<#
--------------------------------------------------------
meanimage
Created By: Andrew Jorquera -JHU
Last Edit: 09/02/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
--------------------------------------------------------
Usage:
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
     $this.sample = $sample
     $this.vers = $this.sample.GetVersion($this.sample.mpath, $this.sample.module, $task[0])

     #$this.sample.info('message')
     #$this.sample.error('error')



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
             $this.processvars = $this.processvars -replace [regex]::escape($this.sample.basepath), $this.processloc
             $this.processvars = $this.processvars -replace [regex]::escape('\\'+$this.sample.project_data.fwpath), ($this.processloc+'\flatw')
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

        #meanimages(main, dd) main = directory with paths document, dd = drive
        $this.sample.info("finished getting mean image")
    }
    <# -----------------------------------------
     returndata
     ------------------------------------------
     Usage: $this.returndata()
    ----------------------------------------- #>
    [void]returndata(){
        #
        $this.sample.info("Return data started")
        
        <#
        copies the image masks and final average image back to the data
        source location

        copies the saved total image and .csv file with metadata(number
        of images and image shape) to the data source
        
        #>
        $this.sample.info("Return data finished")
        #
    }
    <# -----------------------------------------
     cleanup
     cleanup the data directory and return the 
     data to the dpath locations
     ------------------------------------------
     Usage: $this.cleanup()
    ----------------------------------------- #>
    [void]cleanup(){
        #
        $this.sample.info("cleanup started")
        #$this.processvars[1]\flat\*.flt
        #$this.processvars[1]\flat\*.csv
        if ($this.downloadim3ii -eq 1) {
            #delete $this.processvars[0]
        }
        #delete $this.processvars[1]

        #Delete data in working directory---- processvars[1]
        $this.sample.info("cleanup finished")
        #
    }
}
