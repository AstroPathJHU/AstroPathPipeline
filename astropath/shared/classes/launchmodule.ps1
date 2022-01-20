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
        $this.level = 4
        #
        if ($module -match 'batch'){
            $this.sampledefbatch($val[1], $val[0])
        } else {
            $this.sampledefslide($val[1])            

        }
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
            $( & $this.module $this.val $this)  
            $this.output = 0
        } catch {
            $this.error($_.Exception.Message)
            $this.output = 1
        } finally { # end messages
            $this.finish($this.module)
            
        }
        #
    }
    #
}
#
