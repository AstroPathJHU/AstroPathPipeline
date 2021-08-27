#
class launchmodule : mylogger{
    #
    launchmodule(){}
    #
    launchmodule($slideid, $mpath, $module, $val, $teststatus) : base($mpath, $module, $slideid){
        #
        $this.val = $val
        $this.start($module+'-test')
        #
    }
    #
    launchmodule($slideid, $mpath, $module, $val) : base($mpath, $module, $slideid){
        #
        $this.val = $val
        $this.start($module)
        #
        try {
            $( & $module $val $this)   
        } catch {
            $this.error($_.Exception.Message)
        } finally { # end messages
            $this.finish($module)
            }
        #
    }
    #
}
#
