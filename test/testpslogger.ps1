using module .\testtools.psm1
<# -------------------------------------------
 testpslogger
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpslogger : testtools {
    #
    [string]$class = 'logger'
    [string]$module = 'shredxml'
    #
    testpslogger() : base() {
        #
        $this.testloggerconstruction()
        #
        $log = logger $this.mpath $this.module $this.slideid
        Write-Host ($log.project_data | out-string)
        #
        $this.testwritestartmessage($log)
        $this.testeditsampleid($log)
        Write-Host '.'
        #
    }
    #
    [void]testloggerconstruction(){
        #
        Write-Host '.'
        Write-Host 'test [logger] constructors started'
        #
        try {
            logger | Out-Null
        } catch {
            Throw ('[logger] construction with [0] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            logger $this.mpath $this.module | Out-Null
        } catch {
            Throw ('[logger] construction with [2] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
            logger $this.mpath $this.module $this.slideid | Out-Null
        } catch {
            Throw ('[logger] construction with [3] input(s) failed. ' + $_.Exception.Message)
        }
        #
        try {
           logger -mpath $this.mpath -module $this.module -batchid '8' -project '0' | Out-Null
        } catch {
            Throw ('[logger] construction with [4] input(s) failed. ' + $_.Exception.Message)
        }
        #
        Write-Host 'test [logger] constructors finished'
        #
    }
    #
    [void]testwritestartmessage($log){
        #
        Write-Host '.'
        Write-Host 'write to log tests started'
        #
        Write-Host '    write to main log'
        #
        $log.level = 4
        #
        $log.Start('shredxml-test')
        #
        Write-Host '    write to console log'
        #
        $log.level = 8
        #
        $log.Start('shredxml-test')
        #
        $log.level = 12
        #
        Write-Host '    write to main log and console'
        #
        $log.Start('shredxml-test')
        #
        $log.level = 2
        #
        Write-Host '    write to slide log'
        #
        $log.Start('shredxml-test')
        #
        Write-Host 'write to log tests finished'
        #
    }
    #
    [void]testeditsampleid($log){
        #
        Write-Host "."
        Write-Host "test changing the slideid in the logger"
        #
        Write-Host '   '$this.slideid
        Write-Host '        old slideid:' $log.slideid
        Write-Host '        old basepath:' $log.basepath
        Write-Host '        old main log:' $log.mainlog
        Write-Host '        old slide log:' $log.slidelog
        #
        Write-Host '   '$this.slideid2
        $slides = $log.importslideids($this.mpath)
        $log.Sample($this.slideid2, $this.mpath, $slides)
        #
        Write-Host '        new slideid:' $log.slideid
        Write-Host '        new basepath:' $log.basepath
        Write-Host '        new main log:' $log.mainlog
        Write-Host '        new slide log:' $log.slidelog
        Write-Host "test changing the slideid in the logger finished"
    }
    #
}
#
# launch test and exit if no error found
#
[testpslogger]::new() | Out-Null
exit 0
