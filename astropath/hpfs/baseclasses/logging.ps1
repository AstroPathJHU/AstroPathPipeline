<# -------------------------------------------
 write_to_log
 write a message to the log in the specified formating
 projectID;cohortID;slideID;message;timestamp

 Usage: write_to_log -project -cohort -dpath -dname
                     -module -log_string [-log_type -slide_id
                     -message_type -console_output]
 -project: the project ID
 -cohort: the cohort ID
 -dpath: path to the clinical specimen project folder
 -dname: name of the clinical specimen project folder
 -module: the name of the module
 -log_string: string to be output in the message section of the log
 [-log_type]: "master" or "slide" level logging (default is master level)
 [-slide_id]: slide ID for the log must be specified for slide
               level logging (default uses module name)
 [-message_type]: string to be added to log message
                   (ex. "ERROR", "WARNING") (default
                   is "NORMAL" which does not add anything)
 [-console_output]: output log string to the console or not (default is not to)
 ------------------------------------------- #>
 [Flags()] Enum LogLevels {
    IMAGE = 1
    SLIDE = 2
    MAIN = 4
    CONSOLE = 8
    INFO = 16
    WARNING = 32
    ERROR = 64
    STARTED = 128
    FINISHED = 256
 }
#
class mylogger : sampledef {
    [string]$mainlog
    [string]$slidelog
    [int]$level = 6
    [string]$message
    #
    # constructors
    #
    mylogger() : base(){}
    #
    mylogger($slideid, $mpath, $module) : base($slideid, $mpath){
        $this.defpaths($module)
    }
    #
    mylogger($slideid, $mpath, $slides, $module) : base($slideid, $mpath, $slides){
        $this.defpaths($module)
    }
    #
    # define paths
    #
    [void]defpaths($module){
        $this.module = $module
        #
        $this.mainlog = $this.basepath + '\logfiles\' + $this.module + '.log'
        $this.slidelog = $this.basepath + '\' + $this.slideid + '\logfiles\' + $this.slideid + '-' + $this.module + '.log'
        #
        #$this.mainlog = Convert-Path $this.mainlog
        #$this.samplelog = Convert-Path $this.samplelog
    }
    #
    # store level default 
    #
    [void]deflevels([int]$level){
        $this.level = $level
    }
    #
    [string]formatter(
        ){
           $mydate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
           $msg = @($this.Project, $this.Cohort, $this.slideid, $this.message, $mydate) -join ';'
           return  @($msg,"`r`n") -join ''
        }   
    #
    [void]info($msg){
        $this.message = "INFO: " + $msg
        $this.Writelog(2)
    }
    #
    [void]info($msg, $level){
        $this.message = "INFO: " + $msg
        $this.Writelog($level)
    }
    #
    [void]warning($msg){
        $this.message = "WARNING: " + $msg
        $this.Writelog()
    }
    #
    [void]warning($msg, $level){
        $this.message = "WARNING: " + $msg
        $this.Writelog($level)
    }
    #
    [void]error($msg){
        $this.message = "ERROR: " + $msg
        $this.Writelog()
    }
    #
    [void]error($msg, $level){
        $this.message = "ERROR: " + $msg
        $this.Writelog($level)
    }
    #
    [void]start($msg){
        $this.message = "STARTED: " + $msg
        $this.Writelog()
    }
        #
    [void]finish($msg){
        $this.message =  "FINISHED: " + $msg
        $this.Writelog()
    }
    #
    [void]writelog(){
        $this.Writelog($this.level)
    }
    #
    [void]writelog($level){
        #
        if (($level -band [LogLevels]::SLIDE) -eq [LogLevels]::SLIDE){
            $this.PopFile($this.slidelog, $this.formatter())
        }
        #
        if (($level -band [LogLevels]::MAIN) -eq [LogLevels]::MAIN){
            $this.PopFile( $this.mainlog, $this.formatter())
        }
        #
        if (($level -band [LogLevels]::CONSOLE) -eq [LogLevels]::CONSOLE){
            Write-Host $this.formatter()
        }
    }
}

#$a = [MyLogger]::new('LY34', '\\bki04\astropath_processing', 'gitcha')
