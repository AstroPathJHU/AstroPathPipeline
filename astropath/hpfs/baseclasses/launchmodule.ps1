#
class launchmodule : mylogger{
    [array]$val
    #
    launchmodule(){}
    #
    launchmodule($slideid, $mpath, $module, $val) : base($slideid, $mpath, $module){
        #
        # write start messages
        #
        $this.start($module)
        $this.val = $val
        #
        # do module
        #
        try {
            $( & $module $val $this)   
        } catch {
            $this.error($_.Exception.Message)
        } finally { # end messages
            $this.finish($module)
            }
    }
    #
}
#
