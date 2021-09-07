<#
--------------------------------------------------------
meanimage
--------------------------------------------------------
Description
--------------------------------------------------------
Input:
--------------------------------------------------------
Usage:
--------------------------------------------------------
#>
Class meanimage{
    #
    meanimage([array]$task,[launchmodule]$sample){
     $this.sample = $sample
     $this.sample.info('message')
     $this.sample.error('error')

    }
    #
    [void]runmeanimage(){
        $this.downloadim3()
        $this.shreddat()
        $this.getmeanimage()
        $this.returndata()
        $this.cleanup()
    }
    #
}