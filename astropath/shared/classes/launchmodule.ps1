#
class launchmodule : mylogger{
    #
    [int]$output
    #
    launchmodule(){}
    #
    launchmodule($mpath, $module, $val) : base($mpath, $module){
        #
        $this.val = $val
        if ($module -match 'batch'){
            $this.sampledefbatch($val.batchid, $val.project)
            $this.currentlogID = $this.batchid
        } else {
            $this.sampledefslide($val.slideid)
            $this.currentlogID = $this.slideid
        }
        #
        $this.getlogger()
        #
        if ($val.ContainsKey('interactive')) {
            $this.start($this.module)
        } else {
            $this.level = 12
            $this.teststatus = $true
            $this.start($module+'-test')
        }
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
        } finally { # end message
            $this.finish($this.module)
        }
        #
    }
    #
}
#
