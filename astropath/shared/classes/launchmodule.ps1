﻿#
class launchmodule : mylogger{
    #
    [int]$output
    #
    launchmodule(){}
    #
    launchmodule($mpath, $module, $val) : base($mpath, $module){
        #
        $this.val = $val
        $this.level = 12
        #
        if ($module -match 'batch'){
            $this.sampledefbatch($val.batchid, $val.project)
        } else {
            $this.sampledefslide($val.slideid)            

        }
        #
        $this.teststatus = $true
        $this.getlogger()
        $this.start($module+'-test')
        #
    }
    #
    launchmodule($mpath, $module, $slideid, $val) : base($mpath, $module, $slideid){
        #
        $this.val = $val
        $this.executemodule()
        #
    }
    #
    launchmodule($mpath, $module, $batchid, $project, $val) : base($mpath, $module, $batchid, $project){
        #
        $this.val = $val
        $this.executemodule()
        #
    }
    #
    executemodule(){
        #
        $this.start($this.module)
        #
        try {
            initmodule -module $this.module -task $this.val -log $this
            $this.output = 0
        } catch {
            $this.error($_.Exception.Message)
            $this.output = $_
        } finally { # end messages
            $this.finish($this.module)
        }
        #
    }
    #
}
#
